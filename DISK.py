#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 21:12:55 2018

@author: rfm
"""


from copy import deepcopy
import itertools
import pandas as pd
from collections import Counter
import numpy as np
from Bio import SeqIO
#import difflib
#import multiprocessing as mp
#import threading as th
import time
import Levenshtein as levi

import get_varSize_deep
# inexact kNN matching
#import nmslib
import bisect 


def read_fasta_to_dict(fasta_name):
    """ Reads a FASTA file and creates ID to sequence dict.
    
        Requires: filename is the name of a FASTA file.
        Ensures: a dict with IDs as keys and sequences as values.
    """
    
    # read FASTA file
    fasta_file = SeqIO.parse(open(fasta_name),'fasta')
    
    # iterate and create ID to sequence dict
    sequences_dict = {}
    for fasta in fasta_file:
        sequences_dict[fasta.id] = str(fasta.seq)

    return sequences_dict


def mapping_dict(mapping_df,rank):
    """ Creates an ID to taxid dict for specified taxonomic rank.
    
        Requires: mapping_df is a TSV file with taxids and names for each 
        sequence ID. 
        rank is the taxonomic rank that will be considered to create the dict.
        Ensures: a dict with IDs and taxids for the corresponding rank.
    """
    
    # IDs and taxids lists
    ids = list(mapping_df['ids'])
    taxids = list(mapping_df[rank+'_taxids'])
    # create ID to taxid dict
    map_dict = {}
    for m in range(len(ids)):
        map_dict[ids[m]] = taxids[m]

    return map_dict


def kmer_per_id(kmer_size,fasta_dict,map_dict):
    """ Determines distinct k-mers for each taxid.
    
        Requires: kmer_size is the size of the k-mer.
        fasta_dict is a dict with IDs as keys and sequences as values.
        map_dict is a dict with sequence IDs as keys and taxids as values.
        Ensures: dict with taxids as keys and lists of distinct k-mers as
        values.
    """

    kmer_dict = {}
    # define starting position of sliding window
    start = 0
    amb_nucs = ['Y','R','W','S','K','M','D','V','H','B','X','N','-']
    
    # create code that maps taxids to list of sequences ids so that the kmers
    # for a given taxid can be processed sequentially
    
    tax_to_ids = {}
    for i in map_dict:
        tax = map_dict[i]
        if tax in tax_to_ids:
            tax_to_ids[tax].append(i)
        else:
            tax_to_ids[tax] = [i]
    
    for t in tax_to_ids:
        
        ids = tax_to_ids[t]
        
        all_tax_kmers = []
        for i in ids:
            seq = fasta_dict[i]
            
            seq_kmers = []
            # if the initial position plus k-mer size exceeds the sequence length,
            # the while cycle ends.
            while start+kmer_size <= len(seq):
                kmer = seq[start:start+kmer_size]
                # do not consider k-mers with ambiguous bases 'N'
                if not any(amb in kmer for amb in amb_nucs):
                    seq_kmers.append(kmer)
                
                # advance the starting position by 1 base
                # consecutive k-mers overlap by k-1 bases
                start = start + 1
            
            start = 0
            
            all_tax_kmers.append(seq_kmers)
            
        all_tax_kmers = list(itertools.chain.from_iterable(all_tax_kmers))
        all_tax_kmers = list(set(all_tax_kmers))
        
        kmer_dict[t] = all_tax_kmers

    return kmer_dict


def taxa_discriminative_kmers(rank_dict,rank):
    """ Creates the database structure for a given rank with the discriminative 
        k-mers.
    
        For Domain rank, the dict structure will have taxids as keys and
        lists of k-mers as values:
        For ranks below Domain: the dict structure will have a dict with
        the parent nodes taxids as keys and each parent node will have
        a dict with taxids mapping to the lists of k-mers of the rank being 
        determined. parent nodes are used as keys to link with previous level 
        classifications and reduce the search space while relying on higher 
        level classifications that generally are more accurate.
    
        Requires: rank_dict is a dict with taxids as keys and lists of distinct
        k-mers as values.
        rank is the rank of the database/dict structure that will be constructed.
        Ensures: dict with taxids to lists of discriminative k-mers mapping for 
        Domain and with parent node to taxids to discriminative k-mers lists 
        for taxonomic ranks below Domain (Phylum, Class, Order, Family, Genus, 
        Species).
    """
    
    rank_dict_copy = deepcopy(rank_dict)
    rank_dict_discriminative = {}
    
    if rank == 'domain':
        # for each Domain taxid
        for t in rank_dict_copy:
            # get distinct k-mers for the taxid
            current_tax = rank_dict_copy[t]
            other_kmers = []
            # for each Domain taxid
            for k in rank_dict_copy:
                # if the taxid k is different from the current taxid t
                if k != t:
                    # add the k-mers of the taxid k to a list
                    other_kmers.append(rank_dict_copy[k])
            
            # merge all lists of k-mers that are not from the current taxid 
            # into a single list
            all_other_kmers = list(itertools.chain.from_iterable(other_kmers))
            # remove k-mers that are present in the list of k-mers from other
            # taxids from the list of k-mers from the current taxid
            new_current_tax = list(set(current_tax) - set(all_other_kmers))
            
            # add taxid and list with discriminative k-mers just for that taxid 
            # to discriminative k-mers dict
            rank_dict_discriminative[t] = new_current_tax
    
    # for taxonomic ranks below Domain
    else:
        # for each parent taxid
        for p in rank_dict_copy:
            # for each child taxid (current taxid)
            for k in rank_dict_copy[p]:
                # get k-mers for the child taxid (current taxid)
                current_tax = rank_dict_copy[p][k]
                other_kmers = []
                # for each child taxid
                for t in rank_dict_copy[p]:
                    # if the child taxid t is different than the current child
                    # taxid k
                    if k != t:
                        # add child taxid t distinct k-mers to list
                        other_kmers.append(rank_dict_copy[p][t])
                
                # combine all k-mers from child taxids that are not the current
                # child taxid
                all_other_kmers = list(itertools.chain.from_iterable(other_kmers))
                # keep only the k-mers that are discriminative for the current
                # child taxid k
                new_current_tax = list(set(current_tax) - set(all_other_kmers))
                
                # add parent taxid if it is not in dict
                if p not in rank_dict_discriminative:
                    # create dict with parent taxid as key and empty dict as
                    # value
                    rank_dict_discriminative[p] = {}
                    # add current child taxid as key to the empty dict and
                    # list of discriminative k-mers as value for that key
                    rank_dict_discriminative[p][k] = new_current_tax
                else:
                    # if the parent node is in the dict, just add the
                    # child taxid as key and the list of discriminative k-mers
                    # as value
                    rank_dict_discriminative[p][k] = new_current_tax
        
    return rank_dict_discriminative


def invert_dict(tax_dict,rank):
    """ Inverts order of discriminative k-mers dict, keys <--> values for
        the internal dict.
    
        Requires: tax_dict is a dict with taxids mapping to lists of 
        discriminative k-mers for the Domain rank and with parent taxids to
        child taxids to lists of discriminative k-mers for taxonomic ranks
        below Domain.
        Ensures: internal dict with child taxids mapping to discriminative 
        k-mers is inverted and each k-mer becomes a key that maps to its taxid.
        
        {1:{3:['ACT','AGC']}} <--> {1:{'ACT':3,'AGC':3}}
    """
    
    if rank == 'domain':
        inv_dict = {}
        # for each Domain taxid
        for t in tax_dict:
            # get the discriminative k-mers for that taxid
            kmers = tax_dict[t]
            # every k-mer becomes a key in the inverted dict, and the taxid
            # becomes the value for each k-mer
            for k in kmers:
                inv_dict[k] = t
    
    # for taxonomic ranks that are below Domain
    else:
        inv_dict = deepcopy(tax_dict)
        # for each parent taxid
        for p in inv_dict:
            all_dicts = {}
            # for each child taxid
            for t in inv_dict[p]:
                # get discriminative k-mers for that child taxid
                kmers = inv_dict[p][t]
                kmers_dict = {}
                # create inverted dict for that child taxid with each k-mer as
                # key and the child taxid as value
                for k in kmers:
                    kmers_dict[k] = t
                
                # for each child taxid, keep merging the k-mer to taxid mapping
                # dicts so that in the end we have a dict with all discriminative 
                # k-mers for all child taxids of a given parent taxid
                all_dicts = {**all_dicts, **kmers_dict}
            
            # the parent taxid value becomes the inverted dict of all 
            # discriminative k-mers for all all child taxids
            inv_dict[p] = all_dicts
        

    return inv_dict


def import_ncbi_nodes(nodes_file):
    """ Import the nodes.dmp file with the info about the NCBI taxonomy nodes.
    
        Requires: nodes_file is the nodes.dmp file downloaded from the NCBI ftp.
        Ensures: a dataframe with the relevant columns from the nodes.dmp file.
    """
    
    # only read the columns with the nodes IDs, the parent IDs and the 
    # taxonomic rank of the node
    nodes = pd.read_csv(nodes_file,sep='\t',header=None,usecols=[0,2,4])
    
    nodes.columns = ['node_id','parent_id','rank']
    
    # change 'superkingdom' to 'domain'
    nodes['rank'].replace(to_replace='superkingdom', value='domain', inplace=True)
    
    # change column type to numeric
    nodes[['node_id','parent_id']] = nodes[['node_id','parent_id']].apply(pd.to_numeric)

    return nodes


def import_ncbi_names(names_file):
    """ Import the names.dmp file with the info about the NCBI taxonomy names.
    
        Requires: names_file is the names.dmp file downloaded from the NCBI ftp.
        Ensures: a dataframe with the relevant columns from the names.dmp file.
    """
    
    # only read the columns with the nodes IDs, names associated with each node
    # and type of name
    names = pd.read_csv(names_file,sep='\t',header=None,usecols=[0,2,6])
    
    names.columns = ['node_id','name','type']
    
    # select and keep only lines that have scientific names
    mask = names['type'].values == 'scientific name'
    names = names[mask]
    
    # node IDs to numeric
    names['node_id'] = pd.to_numeric(names['node_id'])
    
    return names


def ncbi_tax_dicts(nodes_df,names_df):
    """
    """
    
    # names dict
    nodes_ids = list(names_df['node_id'])
    nodes_names = list(names_df['name'])
    names_dict = dict(zip(nodes_ids, nodes_names))

    # parent dict
    nodes_ids = list(nodes_df['node_id'])
    parents_ids = list(nodes_df['parent_id'])
    parents_dict = dict(zip(nodes_ids, parents_ids))

    # rank dict
    ranks_names = list(nodes_df['rank'])
    ranks_dict = dict(zip(nodes_ids,ranks_names))
        
    return [names_dict,parents_dict,ranks_dict]


def distinct_by_parent(dist_dict,parents_dict,ranks_dict,parent_rank):
    """
    """

    dist_by_parents = {}
    for n in dist_dict:
        parent_id = parents_dict[n]

        base_parent_rank = ranks_dict[parent_id]
        
        while base_parent_rank != parent_rank:
            #print(base_parent_rank,parent_rank)
            prev_id = parent_id
            parent_id = parents_dict[prev_id]
            base_parent_rank = ranks_dict[parent_id]
            if parent_id == 1:
                raise NameError('Node has no parent node!')
        
        if parent_id in dist_by_parents:
            dist_by_parents[parent_id][n] = dist_dict[n]
        elif parent_id not in dist_by_parents:
            dist_by_parents[parent_id] = {n:dist_dict[n]}

    return dist_by_parents


def alter_ncbi_tax(names_df,nodes_df,db_tax,starting_id):
    """
    """

    ranks = ['domain','phylum','class','order','family','genus','species']
    
    new_ids = []
    new_names = []
    new_parents = []
    new_ranks = []
    new_types = []
    
    init_id = starting_id
    
    for r in range(1,len(ranks)):
    
        db_pre = list(db_tax[ranks[r-1]])
        db_pre_taxids = list(db_tax[ranks[r-1]+'_taxids'])
        db_current = list(db_tax[ranks[r]])
        db_current_taxids = list(db_tax[ranks[r]+'_taxids'])

        suffix = ranks[r][0:2]
        
        for g in range(len(db_current)):
            if db_current[g] == '0':
                
                prefix = db_pre[g]
                parent_id = db_pre_taxids[g]
                
                new_name = prefix + ' ' + suffix + '.'
    
                db_current[g] = new_name
                
                if new_name in new_names:
                    db_current_taxids[g] = new_ids[new_names.index(new_name)]
                else:
                    init_id += 1
                    db_current_taxids[g] = init_id
                
                    new_ids.append(init_id)
                    new_names.append(new_name)
                    new_ranks.append(ranks[r])
                    new_types.append('scientific name')
                    new_parents.append(parent_id)
    
        db_tax[ranks[r]] = db_current
        db_tax[ranks[r]+'_taxids'] = db_current_taxids

    new_ncbi_names_ids = list(names_df['node_id']) + new_ids
    new_ncbi_names_names = list(names_df['name']) + new_names
    new_ncbi_names_type = list(names_df['type']) + new_types
    
    names_df = pd.DataFrame()
    names_df['node_id'] = new_ncbi_names_ids
    names_df['name'] = new_ncbi_names_names
    names_df['type'] = new_ncbi_names_type
    
    ranks = ['species','genus','family','order','class','phylum','domain']
    
    nodes_ids = list(nodes_df['node_id'])
    nodes_parents_ids = list(nodes_df['parent_id'])
    for r in range(len(ranks)-1):
        current_tax = list(db_tax[ranks[r]+'_taxids'])
        parent_tax = list(db_tax[ranks[r+1]+'_taxids'])
        for t in range(len(current_tax)):
            taxid = current_tax[t]
            parent_taxid = parent_tax[t]
            if parent_taxid in new_ids and taxid not in new_ids:
            
                taxid_ind = nodes_ids.index(taxid)
                nodes_parents_ids[taxid_ind] = parent_taxid
    
    
    new_ncbi_nodes_ids = nodes_ids + new_ids
    new_ncbi_nodes_parents = nodes_parents_ids + new_parents
    new_ncbi_nodes_ranks = list(nodes_df['rank']) + new_ranks
    
    nodes_df = pd.DataFrame()
    nodes_df['node_id'] = new_ncbi_nodes_ids
    nodes_df['parent_id'] = new_ncbi_nodes_parents
    nodes_df['rank'] = new_ncbi_nodes_ranks
    
  
    return [names_df,nodes_df,db_tax]


def find_taxa_nokmers(disc_dict):
    """
    """
    
    taxa = []
    for k in disc_dict:
        for c in disc_dict[k]:
            if len(disc_dict[k][c]) == 0:
                taxa.append(c)
    
    return taxa


def write_discriminative(discriminative_kmers,filename,rank):
    """
    """
    
    
    if rank != 'domain':
        with open(filename, 'w') as f:
            lines = []
            for p in discriminative_kmers:
                new_lines = [str(p)+','+str(k)+','+str(len(v)) for k,v in discriminative_kmers[p].items()]
                
                lines = lines + new_lines
                
            f.write('parent_taxid,taxid,num_of_discriminative_kmers\n')
                
            for l in lines:
                f.write('%s\n' % l)
                
    else:
        with open(filename, 'w') as f:
            lines = []
            for k, v in discriminative_kmers.items():
                new_line = str(k)+','+str(len(v))
                
                lines.append(new_line)
                
            f.write('taxid,num_of_discriminative_kmers\n')
                
            for l in lines:
                f.write('%s\n' % l)


def write_distinct_tables(distinct_kmers,filename,rank):
    """
    """
    
    
    if rank != 'domain':
        with open(filename, 'w') as f:
        
            for p in distinct_kmers:
                children_dict = distinct_kmers[p]
                children_keys = list(children_dict.keys())
                
                f.write(str(p)+'\n')
                f.write('\t'+'\t'.join(map(str, children_keys)) + '\n')
                
                lines = []
                i = 1
                #top = len(children_keys)
                for c in children_keys:
                    current = children_dict[c]
                    common = []
                    j = 0
                    while len(common) != i:
                        sharedk = len(current) - len((set(current) - set(children_dict[children_keys[j]])))
                        common.append(sharedk)
                        
                        j += 1
                    
                    line = str(c) + '\t' + '\t'.join(map(str, common))
                    lines.append(line)
                    
                    i += 1
                
                for l in lines:
                    f.write('%s \n' % l)
                
                f.write('\n')
    
    else:
        p = 1
        with open(filename, 'w') as f:
                
            f.write(str(p)+'\n')
            
            taxa_keys = list(distinct_kmers.keys())
            f.write('\t'+'\t'.join(map(str, taxa_keys)) + '\n')
                
            lines = []
            i = 1
            for t in taxa_keys:
                current = distinct_kmers[t]
                common = []
                j = 0
                while len(common) != i:
                    sharedk = len(current) - len((set(current) - set(distinct_kmers[taxa_keys[j]])))
                    common.append(sharedk)
                    
                    j += 1
                
                line = str(t) + '\t' + '\t'.join(map(str, common))
                lines.append(line)
                
                i += 1
            
            for l in lines:
                f.write('%s \n' % l)
            
            f.write('\n')


def get_discriminative_dict(kmer_len,fasta_dict,map_dict,
                            crank,prank,parents_dict,ranks_dict):
    """
    """
    
    # get kmer freqs per sequence/id
    distinct_kmers = kmer_per_id(kmer_len,fasta_dict,map_dict)
    
    if crank != 'domain':
        # group taxa distinct kmers by parent taxon before determining discriminative kmers
        dist_kmers_by_parent = distinct_by_parent(distinct_kmers,ncbi_parents_dict,
                                               ncbi_ranks_dict,prank)
        
        # write tables with distinct k-mers shared between taxa with the
        # same parent node
        filename = crank + '_distinct_tables.tsv'
        write_distinct_tables(dist_kmers_by_parent,filename,crank)
        
        # get discriminative kmers for each taxon
        discriminative_kmers = taxa_discriminative_kmers(dist_kmers_by_parent,crank)
        
        # write discrimiantive k-mers table
        filename = crank + '_discriminative_table.csv'
        write_discriminative(discriminative_kmers,filename,crank)
    
        # Invert domain dict to directly match kmer
        # Add parent rank as key before kmers of ranks lower than domain
        discriminative_inv = invert_dict(discriminative_kmers,crank)

        return [discriminative_inv]
    
    elif crank == 'domain':
        
        # write tables with distinct k-mers shared between taxa with the
        # same parent node
        filename = crank + '_distinct_tables.tsv'
        write_distinct_tables(distinct_kmers,filename,crank)
        
        # get discriminative kmers for each taxon
        discriminative_kmers = taxa_discriminative_kmers(distinct_kmers,crank)
        
        # write discrimiantive k-mers table
        filename = crank + '_discriminative_table.csv'
        write_discriminative(discriminative_kmers,filename,crank)
    
        # Invert domain dict to directly match kmer
        # Add parent rank as key before kmers of ranks lower than domain
        discriminative_inv = invert_dict(discriminative_kmers,crank)

        return discriminative_inv


############
# DATABASE #
############

db_name = 'ncbi_complete'
filename = 'ncbi16S.fasta'
kmer_len = 31

# import ncbi nodes and names info
ncbi_nodes = import_ncbi_nodes('nodes.dmp')
ncbi_names = import_ncbi_names('names.dmp')

ids = list(ncbi_names['node_id'])
max_id = max(ids)

# import sequences taxonomy and alter where ranks have no term
db_seqs_map = pd.read_csv('ncbi16S_map.tsv',sep='\t',header=0)

# alter the NCBI taxonomy to include new ranks that were 'not found' and fill incomplete lineages
# some lineages need to be altered to replace invalid names!!!

ncbi_names, ncbi_nodes, db_seqs_map = alter_ncbi_tax(ncbi_names,ncbi_nodes,db_seqs_map,max_id)

# create ncbi dicts
ncbi_names_dict, ncbi_parents_dict, ncbi_ranks_dict = ncbi_tax_dicts(ncbi_nodes,ncbi_names)

del ncbi_nodes
del ncbi_names

# create taxonomic ranks dicts
sp_dict = mapping_dict(db_seqs_map,'species')
ge_dict = mapping_dict(db_seqs_map,'genus')
fa_dict = mapping_dict(db_seqs_map,'family')
or_dict = mapping_dict(db_seqs_map,'order')
cl_dict = mapping_dict(db_seqs_map,'class')
ph_dict = mapping_dict(db_seqs_map,'phylum')
do_dict = mapping_dict(db_seqs_map,'domain')

#np.save('sp_dict.npy', sp_dict)
#np.save('ge_dict.npy', ge_dict)
#np.save('fa_dict.npy', fa_dict)
#np.save('or_dict.npy', or_dict)
#np.save('cl_dict.npy', cl_dict)
#np.save('ph_dict.npy', ph_dict)
#np.save('do_dict.npy', do_dict)

# create generator with sequences and headers
fasta_dict = read_fasta_to_dict(filename)

# Invert domain dict to directly match kmer
# Add parent rank as key before kmers of ranks lower than domain
sp_disc_inv = get_discriminative_dict(kmer_len,fasta_dict,sp_dict,
                            'species','genus',ncbi_parents_dict,ncbi_ranks_dict)

ge_disc_inv = get_discriminative_dict(kmer_len,fasta_dict,ge_dict,
                            'genus','family',ncbi_parents_dict,ncbi_ranks_dict)

fa_disc_inv = get_discriminative_dict(kmer_len,fasta_dict,fa_dict,
                            'family','order',ncbi_parents_dict,ncbi_ranks_dict)

or_disc_inv = get_discriminative_dict(kmer_len,fasta_dict,or_dict,
                            'order','class',ncbi_parents_dict,ncbi_ranks_dict)

cl_disc_inv = get_discriminative_dict(kmer_len,fasta_dict,cl_dict,
                            'class','phylum',ncbi_parents_dict,ncbi_ranks_dict)

ph_disc_inv = get_discriminative_dict(kmer_len,fasta_dict,ph_dict,
                            'phylum','domain',ncbi_parents_dict,ncbi_ranks_dict)

do_disc_inv = get_discriminative_dict(kmer_len,fasta_dict,do_dict,
                            'domain','',ncbi_parents_dict,ncbi_ranks_dict)


#np.save('sp_'+str(kmer_len)+'_disc_inv.npy', sp_disc_inv[0])
#del sp_disc_inv
#
#np.save('ge_'+str(kmer_len)+'_disc_inv.npy', ge_disc_inv[0])
#del ge_disc_inv
#
#np.save('fa_'+str(kmer_len)+'_disc_inv.npy', fa_disc_inv[0])
#del fa_disc_inv
#
#np.save('or_'+str(kmer_len)+'_disc_inv.npy', or_disc_inv[0])
#del or_disc_inv
#
#np.save('cl_'+str(kmer_len)+'_disc_inv.npy', cl_disc_inv[0])
#del cl_disc_inv
#
#np.save('ph_'+str(kmer_len)+'_disc_inv.npy', ph_disc_inv[0])
#del ph_disc_inv
#
#np.save('do_'+str(kmer_len)+'_disc_inv.npy', do_disc_inv)
#del do_disc_inv

##################
# CLASSIFICATION #
##################

#kmer_len = 31
#rank = 'domain'
#rank_kmers = do_disc_inv
#previous_assign = []
#na_mode='greedy'


def inexact_assign(db_dict,seq_kmers,assign_list):
    """ add option to check if the inexact match respects the parent rank assignment...
    """

    # get all database kmers and sort
    db_kmers = list(db_dict.keys())
    db_kmers.sort()
    
    for k in seq_kmers:
        # do not try to search index for kmers with ambiguous
        # nucleotides
        if set(k) == {'A','C','G','T'}:
            # find the index where the kmer would fit into 
            #the list, the adjacent kmers are the most similar
            index = bisect.bisect(db_kmers, k)
            
            # calculate edit distance - Levenshtein
            # for the kmer that precedes the index
            if index != 0:
                levi_dist = levi.distance(db_kmers[index-1],k)
                if levi_dist <= 1:
                    assign_list.append(db_dict[db_kmers[index-1]])
            
            # if the most appropriate index is at the start
            # of the list (0), calculate distance for that index
            # since there is no preceding index
            elif index == 0:
                levi_dist = levi.distance(db_kmers[index],k)
                if levi_dist <= 1:
                    assign_list.append(db_dict[db_kmers[index]])

    return assign_list


def calc_confidence(taxon1_hits, taxon2_hits):
    """
    """
    
    confidence = taxon1_hits / (taxon1_hits + taxon2_hits)
    
    return confidence


def calc_gamma(taxon1_hits, taxon2_hits, total_kmers):
    """
    """

    gamma = (taxon1_hits + taxon2_hits) / total_kmers
    
    return gamma


def assign_na(db_kmers,seq_kmers,na_mode):
    """
    """
    
    assign = []   
    if na_mode == 'greedy':
        for b in db_kmers:
            crank = db_kmers[b]
            assign = [crank[k] for k in seq_kmers if k in crank]
            if len(assign) > 0:
                break
    
    elif na_mode == 'full':
        for b in db_kmers:
            crank = db_kmers[b]
            inter_hits = [crank[k] for k in seq_kmers if k in crank]
            if len(inter_hits) > 0:
                assign = assign + inter_hits
                
    return assign



def classify_seqs(kmer_len,fasta_dict,rank,rank_kmers,previous_assign,na_mode='greedy'):
    """
    Divides a sequence into k-mers of specified length and classifies sequences
    based on exact matches of discriminative k-mers stored in a pre-constructed
    database, with the option to classify based on inexact matches if there are
    no exact matches.
    
    Requires: 
        kmer_len: the k-mer size used to divide the sequences to
                  classify and also the k-mer size used to construct the database;
        fasta_dict: a dict with sequences ids as keys and strings
                    representing the DNA sequences as values;
        rank: the taxonomic rank to classify at;
        rank_kmers: the database dict structure with the discriminative k-mers
                    for each taxa of the specified rank;
        previous_assign: a dict with sequences ids as keys and 7-element tuples
                         as values. The 2nd element in the tuple is the taxon
                         determined for the sequence at the parent rank and is
                         used to select the child taxa that are considered
                         valid based on the parent rank assignment;
        na_mode: the classification mode used to classify sequences that have
                 no parent rank assignment. 'greedy' will start iterating over
                 the taxa and assign the sequence to the first taxon that has
                 hits. 'full' will iterate over all taxa, store all hits and 
                 assign the sequence to the taxon with most hits.
              
    Ensures:
        A dict with sequences ids as keys and 7-element tuples as values. 
        The 7 elements in the tuple are:
            The total length of the sequence; 
            The first taxon to which the sequence was assigned; 
            The number of discriminative k-mers found in the database for that 
            first assignment; 
            The second taxon with most hits;
            The number of discriminative k-mers found for the 2nd taxon; 
            A confidence score that represents the proportion of discriminative
            k-mers found for the first taxon over all discriminative k-mers 
            found for both assignments.
            A gamma score that represents the proportion of discriminative 
            k-mers found over all k-mers determined for the sequence.
    """
    
    # dict to store id to assignment
    assignments = {}
    
    # for each id and sequence
    for idt, seq in fasta_dict.items():
        
        # decompose the sequence into all kmers overlapping k-1 bases
        kmer_list = [seq[i:i+kmer_len] for i in range(len(seq) - kmer_len + 1)]
        
        # domain assignment is different because it is the first assignment
        # and there are no previous assignments and the database dict has
        # no parent taxids as keys.
        if rank == 'domain':
            
            # for each kmer in the sequence check if it is in the database
            assign = [rank_kmers[k] for k in kmer_list if k in rank_kmers]
            
            if len(assign) == 0:
                    
                assign = inexact_assign(rank_kmers,kmer_list,assign)

        else:
            
            # get the previous assignment for the current sequence id
            parent_rank = previous_assign[idt][1]
            
            # if the sequence was classified at the parent rank
            if parent_rank != 'not found':
                # get the dict with the child taxa
                parent_dict = rank_kmers[parent_rank]
                
                # search for each kmer in the child taxa dict
                assign = [parent_dict[k] for k in kmer_list if k in parent_dict]
                
                # find closest kmer and assign taxon of that kmer
                # maximizes recall but lowers precision
                if len(assign) == 0:
                    
                    assign = inexact_assign(parent_dict,kmer_list,assign)

            else:
                
                assign = assign_na(rank_kmers,kmer_list,na_mode)
        
        if len(assign) == 0:
            assignments[idt] = (len(seq),'NA','NA',
                                'NA','NA','NA','NA')
        else:
            # choose the two taxa with most hits
            cnt = Counter(assign)
            
            cnt_tulist = cnt.most_common()
            
            taxon1, taxon1_hits = cnt_tulist[0][0], cnt_tulist[0][1]
            
            try:
                taxon2, taxon2_hits = cnt_tulist[1][0], cnt_tulist[1][1]
            except:
                taxon2 = 'NA'
                taxon2_hits = 0
            
            # calculate confidence and gamma like CLARK
            confidence = calc_confidence(taxon1_hits, taxon2_hits)
            gamma = calc_gamma(taxon1_hits, taxon2_hits, len(kmer_list))
            
            assignments[idt] = (len(seq),taxon1,taxon1_hits,
                                taxon2,taxon2_hits,confidence,gamma)

    return assignments



filename = 'ncbi16S.fasta'
kmer_len = 31
na_assign = 'greedy'

fasta_dict = read_fasta_to_dict(filename)


#start_time = time.time()

# classify at domain
do_disc_inv = np.load('do_31_disc_inv.npy').item()

start_time = time.time()
domain_assignments = classify_seqs(kmer_len,fasta_dict,'domain',do_disc_inv,
                                   [],na_assign)
print("--- %s seconds ---" % (time.time() - start_time))

del do_disc_inv

# classify at phylum
ph_disc_inv = np.load('ph_31_disc_inv.npy').item()
start_time = time.time()
phylum_assignments = classify_seqs(kmer_len,fasta_dict,'phylum',ph_disc_inv,
                                   domain_assignments,na_assign)
print("--- %s seconds ---" % (time.time() - start_time))
del ph_disc_inv

# classify at class
cl_disc_inv = np.load('cl_31_disc_inv.npy').item()
start_time = time.time()
class_assignments = classify_seqs(kmer_len,fasta_dict,'class',cl_disc_inv,
                                  phylum_assignments,na_assign)
print("--- %s seconds ---" % (time.time() - start_time))
del cl_disc_inv

# classify at order
or_disc_inv = np.load('or_31_disc_inv.npy').item()
start_time = time.time()
order_assignments = classify_seqs(kmer_len,fasta_dict,'order',or_disc_inv,
                                  class_assignments,na_assign)
print("--- %s seconds ---" % (time.time() - start_time))
del or_disc_inv

# classify at family
fa_disc_inv = np.load('fa_31_disc_inv.npy').item()
start_time = time.time()
family_assignments = classify_seqs(kmer_len,fasta_dict,'family',fa_disc_inv,
                                   order_assignments,na_assign)
print("--- %s seconds ---" % (time.time() - start_time))
del fa_disc_inv

# classify at genus
ge_disc_inv = np.load('ge_31_disc_inv.npy').item()
start_time = time.time()
genus_assignments = classify_seqs(kmer_len,fasta_dict,'genus',ge_disc_inv,
                                  family_assignments,na_assign)
print("--- %s seconds ---" % (time.time() - start_time))
del ge_disc_inv

# classify at species
sp_disc_inv = np.load('sp_31_disc_inv.npy').item()
start_time = time.time()
species_assignments = classify_seqs(kmer_len,fasta_dict,'species',sp_disc_inv,
                                    genus_assignments,na_assign)
print("--- %s seconds ---" % (time.time() - start_time))
del sp_disc_inv

#print("--- %s seconds ---" % (time.time() - start_time))

def write_assignments(assignments,filename):
    """
    """
    
    with open(filename, 'w') as f:
        
        f.write('id,seq_len,taxon1,tx1_hits,taxon2,tx2_hits,confidence,gamma\n')
        
        lines = [k+','+','.join(map(str, v)) for k, v in assignments.items()]
        
        for l in lines:
            f.write("%s\n" % l)
    

write_assignments(domain_assignments,"domain_assignments.csv")
write_assignments(phylum_assignments,"phylum_assignments.csv")
write_assignments(class_assignments,"class_assignments.csv")
write_assignments(order_assignments,"order_assignments.csv")
write_assignments(family_assignments,"family_assignments.csv")
write_assignments(genus_assignments,"genus_assignments.csv")
write_assignments(species_assignments,"species_assignments.csv")


#assignments = domain_assignments
#true_taxa = do_dict

def calculate_metrics(assignments,true_taxa):
    """
    """
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    total_assign = len(assignments)
    neg_checker = {}
    for idt, tup in assignments.items():
        taxon1 = int(tup[1]) if tup[1] != 'NA' else tup[1]
        true_taxon = true_taxa[idt]
        try:
            if taxon1 == true_taxon:
                tp += 1
            elif taxon1 != true_taxon and taxon1 != 'not found':
                fp += 1
                neg_checker[idt] = [taxon1,true_taxon]
            elif taxon1 == 'not found':
                fn += 1
                neg_checker[idt] = [taxon1,true_taxon]
        except:
            total_assign -= 1
            
    # precision
    prec = tp/(tp+fp)
    
    # recall
    rec = tp/(tp+fn)
    
    # accuracy
    acc = (tp+tn)/(tp+tn+fp+fn)
    
    return [prec,rec,acc,total_assign,neg_checker]


def import_assignments(input_file):
    """
    """
    
    with open(input_file, 'r') as file:
        assignments = file.readlines()
        assignments = assignments[1:]
        assignments = [e.strip() for e in assignments]
        assignments = [l.split(',') for l in assignments]
    
    tax_assignments = {}
    for l in assignments:
        tax_assignments[l[0]] = l[1:]
    
    return tax_assignments
    
    

# import true_classifications
do_dict = np.load('do_dict.npy').item()
# import assignments
domain_assignments = import_assignments('domain_assignments.csv')
do_metrics = calculate_metrics(domain_assignments,do_dict)

ph_dict = np.load('ph_dict.npy').item()
phylum_assignments = import_assignments('phylum_assignments.csv')
ph_metrics = calculate_metrics(phylum_assignments,ph_dict)

cl_dict = np.load('cl_dict.npy').item()
class_assignments = import_assignments('class_assignments.csv')
cl_metrics = calculate_metrics(class_assignments,cl_dict)

or_dict = np.load('or_dict.npy').item()
order_assignments = import_assignments('order_assignments.csv')
or_metrics = calculate_metrics(order_assignments,or_dict)

fa_dict = np.load('fa_dict.npy').item()
family_assignments = import_assignments('family_assignments.csv')
fa_metrics = calculate_metrics(family_assignments,fa_dict)
# metrics aren't perfect because some of the sequences at the family level
# weren't classified. Those sequences probably have no kmers that could be matched
# to the database. Nonetheless, most sequences of the same families could be identified.

ge_dict = np.load('ge_dict.npy').item()
genus_assignments = import_assignments('genus_assignments.csv')
ge_metrics = calculate_metrics(genus_assignments,ge_dict)
# same as with family rank. The genera in the same family as Streptomyces are 
# the most affected and most sequences that are not classified are from those genera

sp_dict = np.load('sp_dict.npy').item()
species_assignments = import_assignments('species_assignments.csv')
sp_metrics = calculate_metrics(species_assignments,sp_dict)
# classification is good at species but 182 species cannot be identified because
# no discriminative kmers could be retained for those species and because the discriminative
# kmers retained also cannot identify every single sequence from certain species.


#############
# Dash test #
#############

import dash
import dash_core_components as dcc
import dash_html_components as html


app = dash.Dash()

kmers = pd.read_csv('species_discriminative_table.csv')


from plotly import tools
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.figure_factory as ff


hist_data = [list(kmers['num_of_discriminative_kmers'])]
group_labels = ['kmers']

fig = ff.create_distplot(hist_data,group_labels,curve_type='kde', show_rug=False,
                                    show_hist=False,histnorm='probability density')

plot(fig, filename='dash_test.html', auto_open=True)














