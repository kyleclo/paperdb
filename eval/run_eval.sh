#!/bin/bash

# Run evaluation on oracle predictions
python eval/score.py results/oracle.jsonl

# Run evaluation on random predictions
python eval/score.py results/random.jsonl
