# for file in Dataset/TAQ/trades/*.tar.gz; do
#     tar -xzvf "$file" -C Impact_Model/trades
# done

for file in Dataset/TAQ/quotes/*.tar.gz; do
    tar -xzvf "$file" -C Impact_Model/quotes
done