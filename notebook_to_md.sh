#!/bin/bash

# Convert ipynb files to markdown.
jupyter nbconvert --to notebook --execute /example_scripts/notebooks/runnable_notebooks/*.ipynb --allow-errors --output-dir /example_scripts/notebooks/runnable_notebooks/executed/
jupyter nbconvert --to markdown /example_scripts/notebooks/runnable_notebooks/executed/*.ipynb --output-dir /example_scripts/notebooks/markdown/
rm -rf ./example_scripts/notebooks/runnable_notebooks/executed  # Delete temp executed notebook dir.
rm -rf ./example_scripts/notebooks/runnable_notebooks/*.nc  # Delete output nc files.
mkdir /example_scripts/notebooks/markdown_images/ # image folder

# Loop through generated markdown files.
for FILE in $(find ./example_scripts/notebooks/markdown -name '*.md'); do 
    # Get filename information.
    fullname=$(basename $FILE)
    filename=${fullname%.*}
    formatted_filename=${filename//[_]/ }   # Strip out underscores.
    formatted_filename=${formatted_filename^}   # Capitalize.
    # Generate Hugo header for Markdown files.
    read -r -d '' VAR <<-EOM
---
    title: "$formatted_filename"
    linkTitle: "$formatted_filename"
    weight: 5

    description: >
        $formatted_filename example.
---
EOM
    # Echo hugo header to beginning of generated md file.
    echo "$VAR" | cat - $FILE > temp && mv temp $FILE
    sed -i "s+${filename}_files/+/COAsT/${filename}_files/+g" $FILE
    mv  /example_scripts/notebooks/markdown/${filename}_files /example_scripts/notebooks/markdown_images/
done
