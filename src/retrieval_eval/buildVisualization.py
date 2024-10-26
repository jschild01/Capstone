import pandas as pd
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go

def generate_combined_line_charts():
    # Directory path setup
    retriever_eval_dir = os.path.dirname(__file__)
    
    # Load the data
    data_path = os.path.join(retriever_eval_dir, "RetrievalEvaluation.xlsx")
    df = pd.read_excel(data_path)

    # Rename accuracy column for easier reference
    df.rename(columns={'Accuracy (% Docs Correct Out of 13 Q/As)': 'Accuracy'}, inplace=True)
    df = df.sort_values(by='Top_k', ascending=False)

    # Replace model names
    df['Model'] = df['Model'].replace('titan', 'amazon.titan-embed-text-v2:0', regex=True)
    df['Model'] = df['Model'].replace('instructor', 'hkunlp/instructor-xl', regex=True)
    df['Model'] = df['Model'].replace('mini', 'all-MiniLM-L6-v2', regex=True)

    # Define color map using blue, green, and orange
    color_cycle = ['#1f77b4', '#2ca02c', '#ff7f0e']  # Blue, Green, Orange
    unique_models = df['Model'].unique()
    color_map = {model: color_cycle[i % len(color_cycle)] for i, model in enumerate(unique_models)}

    # Set up a figure with 3x2 layout (3 on top, 2 below)
    unique_chunk_sizes = df['Chunk Size'].unique()
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()  # Flatten axes array for easy iteration

    for i, chunk_size in enumerate(unique_chunk_sizes[:5]):  # Limit to 5 plots if there are more chunk sizes
        ax = axes[i]
        subset = df[df['Chunk Size'] == chunk_size]

        for model in subset['Model'].unique():
            model_data = subset[subset['Model'] == model]
            ax.plot(model_data['Top_k'], model_data['Accuracy'], marker='o', label=model, color=color_map[model])
        
        # Subplot formatting
        ax.set_title(f'Chunk Size {chunk_size}')
        ax.set_xlabel('Top_k')
        ax.set_ylabel('Accuracy')
        ax.legend(title='Model', fontsize='small')
        ax.grid(True)

    # Hide the unused subplot(s)
    for j in range(len(unique_chunk_sizes), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and save the combined plot
    plt.tight_layout()
    combined_plot_filename = os.path.join(retriever_eval_dir, 'SampleData_AccEval_linecharts.png')
    plt.savefig(combined_plot_filename)
    plt.close()  # Close to free up memory



def get_lowest_top_k_per_chunk(group):
    group_sorted = group.sort_values(by='Top_k').drop_duplicates(subset=['Chunk Size'], keep='first')
    return group_sorted
    
def generate_tables():
    # Load the Excel file
    retreiver_eval_dir = os.path.dirname(__file__)
    
    # Get CSV file
    data_path = os.path.join(retreiver_eval_dir, "RetrievalEvaluation.xlsx")
    df = pd.read_excel(data_path)

    # Retrieve the best accuracies, filter to keep only the lowest top_k per chunk size, and format the top_k values
    best_accuracies_filtered = df.groupby('Model').apply(
        lambda x: get_lowest_top_k_per_chunk(x[x['Accuracy (% Docs Correct Out of 13 Q/As)'] == x['Accuracy (% Docs Correct Out of 13 Q/As)'].max()])
    ).reset_index(drop=True)
    best_accuracies_filtered['Top_k'] = best_accuracies_filtered.apply(
        lambda row: f"{row['Top_k']}-{row['Chunk Size']}", axis=1
    )
    best_accuracies_final = best_accuracies_filtered.groupby('Model').agg({
        'Top_k': lambda x: ', '.join(sorted(x, key=lambda s: int(s.split('-')[0]))),
        'Accuracy (% Docs Correct Out of 13 Q/As)': 'first'
    }).reset_index()
    best_accuracies_final.columns = ['Model', 'Top_k', 'Accuracy (% Docs Correct Out of 13 Q/As)']
    best_accuracies_final['Rank'] = 'Accuracy (% Docs Correct Out of 13 Q/As)'

    # Retrieve the second-best accuracies for each model
    second_best_accuracies = df.groupby('Model').apply(
        lambda x: x[x['Accuracy (% Docs Correct Out of 13 Q/As)'] < x['Accuracy (% Docs Correct Out of 13 Q/As)'].max()].nlargest(1, 'Accuracy (% Docs Correct Out of 13 Q/As)')
    ).reset_index(drop=True)
    second_best_accuracies['Top_k'] = second_best_accuracies.apply(
        lambda row: f"{row['Top_k']}-{row['Chunk Size']}", axis=1
    )
    second_best_accuracies = second_best_accuracies[['Model', 'Top_k', 'Accuracy (% Docs Correct Out of 13 Q/As)']]
    second_best_accuracies.columns = ['Model', 'Top_k', 'Accuracy (% Docs Correct Out of 13 Q/As)']
    second_best_accuracies['Rank'] = 'Accuracy (% Docs Correct Out of 13 Q/As)'

    # Combine best and second-best accuracies into a single table and sort by accuracy
    final_combined_accuracies = pd.concat([best_accuracies_final, second_best_accuracies], ignore_index=True)
    final_combined_accuracies['Accuracy (% Docs Correct Out of 13 Q/As)'] = final_combined_accuracies['Accuracy (% Docs Correct Out of 13 Q/As)'].apply(lambda x: f"{x * 100:.2f}%")
    final_combined_accuracies = final_combined_accuracies.sort_values(by='Accuracy (% Docs Correct Out of 13 Q/As)', ascending=False).reset_index(drop=True)

    # final reformats
    final_combined_accuracies = final_combined_accuracies.drop(columns=['Rank'])
    final_combined_accuracies = final_combined_accuracies.rename(columns={'Top_k': 'Params (Topk-ChunkSize)'})

    # generate image
    fig, ax = plt.subplots(figsize=(10, len(final_combined_accuracies) * 0.5))  # Adjust figure size as needed
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=final_combined_accuracies.values, colLabels=final_combined_accuracies.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Scale table size

    # Automatically set column widths based on content length
    for i, col in enumerate(final_combined_accuracies.columns):
        max_len = max(len(str(val)) for val in final_combined_accuracies[col])  # Get max length in the column
        table.auto_set_column_width(i)

    # Set header style: bold text and light blue background
    for col in range(len(final_combined_accuracies.columns)):
        header_cell = table[0, col]  # Access each header cell by row and column
        header_cell.set_text_props(weight='bold')
        header_cell.set_facecolor('#ADD8E6')  # Light blue color

    # save image
    output_path = os.path.join(retreiver_eval_dir, "SampleData_AccEval_table.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Table image saved to {output_path}")


generate_combined_line_charts()
generate_tables()