#!/bin/bash

# set directory
directory=$1

# Convert ipynb files to markdown.
echo "starting notebook execute for ${directory}"
jupyter nbconvert --to notebook --execute example_scripts/notebooks/runnable_notebooks/${directory}/*.ipynb --allow-errors --output-dir example_scripts/notebooks/runnable_notebooks/executed/${directory}/
echo "starting notebook convert"
jupyter nbconvert --to markdown example_scripts/notebooks/runnable_notebooks/executed/${directory}/*.ipynb --output-dir example_scripts/notebooks/markdown/${directory}/
echo "starting clean up (rm)"
rm -rf example_scripts/notebooks/runnable_notebooks/executed/${directory}  # Delete temp executed notebook dir.
rm -rf example_scripts/notebooks/runnable_notebooks/${directory}/*.nc  # Delete output nc files.

mkdir -p example_scripts/notebooks/markdown_images/${directory}/ # image folder

echo "starting loop"
# Loop through generated markdown files.
for FILE in $(find example_scripts/notebooks/markdown/${directory} -name '*.md'); do
    # Get filename information.
    fullname=$(basename $FILE)
    filename=${fullname%.*}
    echo ${filename}
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
    [ -d /example_scripts/notebooks/markdown/${directory}/${filename}_files ] && mv  example_scripts/notebooks/markdown/${directory}/${filename}_files example_scripts/notebooks/markdown_images/${directory}/
    echo "script is all done"
done

