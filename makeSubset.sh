#!/bin/bash

src_dir=$1
des_dir=$2
limit=$3

if [ ! -d des_dir ]; 
then
	mkdir -p "des_dir"
fi

ls "$src_dir" | head -$limit | xargs -I {} cp "$src_dir/{}" "$des_dir/"
