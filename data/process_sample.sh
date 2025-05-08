#!/bin/bash

# Check if file is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi

# Extract numbers, replace "0." with "0", join with commas and spaces
cat "$1" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g' | 
sed 's/ / ▪/g' |  # Mark word boundaries
sed 's/▪0\. /▪0 /g' |  # Replace exactly "0." with "0" only when it's a complete token
sed 's/▪//g' |  # Remove markers
tr ' ' '\n' | grep -v '^$' | 
# Format with commas and insert newlines every 20 numbers
awk '
  BEGIN { count = 0; line = ""; }
  {
    if (count > 0) line = line ", ";
    line = line $0;
    count++;
    if (count == 20) {
      print line;
      count = 0;
      line = "";
    }
  }
  END {
    if (count > 0) print line;
  }
'
