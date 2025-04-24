import ipywidgets as widgets
from IPython.display import display, Markdown

# Constants and assumptions
TOKENS_PER_USER = 10  # Desired tokens/sec per user
TOKENS_PER_SECOND_PER_H100 = {
    # Scale inversely with model size
    'FP16': lambda params: 1500 * (70 / params),
    # Optimistic throughput with quantization
    'INT4': lambda params: 2500 * (70 / params),
}


def estimate_users(throughput, target_tok_per_user):
    return int(throughput / target_tok_per_user)


def update_output(change=None):
    params = param_slider.value
    gpus = gpu_slider.value
    precision = precision_dropdown.value
    tok_per_user = token_slider.value

    total_throughput = TOKENS_PER_SECOND_PER_H100[precision](params) * gpus
    user_capacity = estimate_users(total_throughput, tok_per_user)

    output_area.value = f"### Estimated User Capacity\n" \
                        f"**Model size:** {params}B\n" \
                        f"**# GPUs (H100):** {gpus}\n" \
                        f"**Precision:** {precision}\n" \
                        f"**Throughput:** ~{int(total_throughput)} tokens/sec\n" \
                        f"**Max users @ {tok_per_user} tok/s per user:** **~{user_capacity} users**"


# UI Widgets
param_slider = widgets.IntSlider(
    value=70, min=1, max=175, step=1, description='Model (B):')
gpu_slider = widgets.IntSlider(
    value=1, min=1, max=16, step=1, description='GPUs (H100):')
precision_dropdown = widgets.Dropdown(
    options=['FP16', 'INT4'], value='FP16', description='Precision:')
token_slider = widgets.IntSlider(
    value=10, min=1, max=50, step=1, description='Tok/s per user:')
output_area = widgets.HTML()

# Set up interactivity
for widget in [param_slider, gpu_slider, precision_dropdown, token_slider]:
    widget.observe(update_output, names='value')

# Initial update
update_output()

# Display widgets
display(widgets.VBox([param_slider, gpu_slider,
        precision_dropdown, token_slider, output_area]))
