import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tabulate import tabulate
from PyPDF2 import PdfReader

# Define paths
data_dir = "data/CUAD_v1/full_contract_pdf"
train_parts = ["Part_I", "Part_III"]
test_part = "Part_II"

# Function to get all PDF files in a directory and its subdirectories


def get_pdf_files(directory):
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

# Function to get category from file path


def get_category(file_path):
    parts = file_path.split(os.sep)
    # The category should be the directory name before the PDF file
    for i, part in enumerate(parts):
        if part in ["Part_I", "Part_II", "Part_III"]:
            if i+1 < len(parts)-1:  # Ensure there's a category directory
                return parts[i+1]
    return "unknown"

# Function to get file size in KB


def get_file_size(file_path):
    return os.path.getsize(file_path) / 1024  # Convert bytes to KB

# Function to get number of pages in PDF


def get_page_count(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            return len(pdf.pages)
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return 0


# Collect training and test files
train_files = []
for part in train_parts:
    part_path = os.path.join(data_dir, part)
    if os.path.exists(part_path):
        train_files.extend(get_pdf_files(part_path))

test_files = []
test_path = os.path.join(data_dir, test_part)
if os.path.exists(test_path):
    test_files.extend(get_pdf_files(test_path))

print(f"Found {len(train_files)} training files and {len(test_files)} test files")

# Analyze categories
train_categories = [get_category(f) for f in train_files]
test_categories = [get_category(f) for f in test_files]

train_cat_counts = pd.Series(train_categories).value_counts()
test_cat_counts = pd.Series(test_categories).value_counts()

# Merge to get all categories
all_categories = pd.concat(
    [train_cat_counts, test_cat_counts], axis=1, sort=True).fillna(0)
all_categories.columns = ['Train', 'Test']
all_categories['Train'] = all_categories['Train'].astype(int)
all_categories['Test'] = all_categories['Test'].astype(int)

# Calculate p-values using chi-square test for each category
p_values = []
for category in all_categories.index:
    observed = [all_categories.loc[category, 'Train'],
                all_categories.loc[category, 'Test']]
    expected = [sum(observed) * (len(train_files) / (len(train_files) + len(test_files))),
                sum(observed) * (len(test_files) / (len(train_files) + len(test_files)))]

    # Only perform test if we have enough samples
    if sum(observed) > 5 and min(observed) > 0:
        _, p_value = stats.chisquare(observed, expected)
    else:
        p_value = np.nan

    p_values.append(p_value)

all_categories['p-value'] = p_values

# Analyze file sizes and page counts
train_sizes = [get_file_size(f) for f in train_files]
test_sizes = [get_file_size(f) for f in test_files]

print("Counting pages in training files...")
train_pages = [get_page_count(f) for f in train_files]
print("Counting pages in test files...")
test_pages = [get_page_count(f) for f in test_files]

# Statistical tests
size_t_stat, size_p_value = stats.ttest_ind(
    train_sizes, test_sizes, equal_var=False)
pages_t_stat, pages_p_value = stats.ttest_ind(
    train_pages, test_pages, equal_var=False)

# Create a summary table
summary_table = pd.DataFrame({
    'Metric': [
        'Number of files',
        'Number of categories',
        'Mean file size (KB)',
        'Median file size (KB)',
        'Mean page count',
        'Median page count',
        'File size p-value',
        'Page count p-value'
    ],
    'Training': [
        len(train_files),
        len(train_cat_counts),
        np.mean(train_sizes),
        np.median(train_sizes),
        np.mean(train_pages),
        np.median(train_pages),
        size_p_value,
        pages_p_value
    ],
    'Test': [
        len(test_files),
        len(test_cat_counts),
        np.mean(test_sizes),
        np.median(test_sizes),
        np.mean(test_pages),
        np.median(test_pages),
        size_p_value,
        pages_p_value
    ]
})

# Display results
print("\nSummary Statistics:")
print(tabulate(summary_table, headers='keys', tablefmt='grid', showindex=False))

print("\nCategory Distribution:")
print(tabulate(all_categories, headers='keys', tablefmt='grid'))

# Visualize category distribution
plt.figure(figsize=(12, 8))
all_categories[['Train', 'Test']].plot(kind='bar')
plt.title('Category Distribution in Training and Test Sets')
plt.ylabel('Number of Documents')
plt.xlabel('Category')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Visualize file size distribution
plt.figure(figsize=(10, 6))
plt.hist(train_sizes, alpha=0.5, label='Training', bins=30)
plt.hist(test_sizes, alpha=0.5, label='Test', bins=30)
plt.title('File Size Distribution')
plt.xlabel('File Size (KB)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Visualize page count distribution
plt.figure(figsize=(10, 6))
plt.hist(train_pages, alpha=0.5, label='Training', bins=30)
plt.hist(test_pages, alpha=0.5, label='Test', bins=30)
plt.title('Page Count Distribution')
plt.xlabel('Number of Pages')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Print categories with significant differences (p < 0.05)
significant_categories = all_categories[all_categories['p-value'] < 0.05]
if not significant_categories.empty:
    print("\nCategories with significant differences (p < 0.05):")
    print(tabulate(significant_categories, headers='keys', tablefmt='grid'))
else:
    print("\nNo categories with significant differences found (p < 0.05)")
