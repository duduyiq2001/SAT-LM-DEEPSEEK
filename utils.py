"""
Utility functions for file operations and data manipulation.

This module provides helper functions for common tasks like
reading and writing JSON files, flattening nested lists,
and other general utilities needed across the codebase.
"""

import os
import argparse
import sys
import json
import re

import numpy as np

from abc import ABC
from collections import namedtuple, Counter, OrderedDict
from os.path import join
import itertools

def read_jsonline(fname):
    """Read a JSONL file, filtering out commented or empty lines.
    
    Args:
        fname (str): Path to the JSONL file
        
    Returns:
        list: List of parsed JSON objects
    """
    with open(fname) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        lines = [x for x in lines if (not x.startswith("#")) and len(x) > 0]
    return [json.loads(x) for x in lines]

def read_json(fname):
    """Read a JSON file.
    
    Args:
        fname (str): Path to the JSON file
        
    Returns:
        object: Parsed JSON object
    """
    with open(fname) as f:
        return json.load(f)

def dump_json(obj, fname, indent=None):
    """Write an object to a JSON file.
    
    Args:
        obj: Object to serialize
        fname (str): Path to the output file
        indent (int, optional): Indentation level for pretty-printing
        
    Returns:
        object: Result of json.dump
    """
    with open(fname, 'w', encoding='utf-8') as f:
        return json.dump(obj, f, indent=indent)

def flatten_nested_list(x):
    """Flatten a nested list into a single list.
    
    Args:
        x (list): List of lists
        
    Returns:
        list: Flattened list
    """
    return list(itertools.chain(*x))