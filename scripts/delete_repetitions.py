import csv

input_file = "generated_non_extremist.csv"
output_file = "generated_non_extremist.csv"

with open(input_file, mode="r", newline="", encoding="utf-8") as infile:
    reader = csv.reader(infile)
    rows = list(reader)

unique_rows = []
seen = set()
for row in rows:
    row_tuple = tuple(row)  
    if row_tuple not in seen:
        seen.add(row_tuple)
        unique_rows.append(row)

with open(output_file, mode="w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(unique_rows)

print(f"Duplicates removed. Clean file saved as '{output_file}'.")
