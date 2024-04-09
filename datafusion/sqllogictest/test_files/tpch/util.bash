#!/bin/bash

# Check if a file name is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# The file to be read, provided as the first argument
filename="$1"

# Check if the file exists
if [ ! -f "$filename" ]; then
    echo "File not found!"
    exit 2
fi

# Initialize a line number counter
line_number=1

# Read the file line by line
while IFS= read -r line; do
    # Prepend the line number to each line and print
    echo "${line_number}: $line"
    # Increment the line number
    ((line_number++))
done < "$filename"
