from langchain_community.llms import LlamaCpp


def load_model(gguf_model_path: str) -> LlamaCpp:
    """
    Load the model

    Parameters:
    gguf_model_path (str): path to the gguf model

    Returns:
    llm (LlamaCpp): LlamaCpp object

    """
    n_gpu_layers = (
        -1
    )  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=gguf_model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        verbose=True,  # Verbose is required to pass to the callback manager
        n_ctx=15000,
        temperature=0.0,
        seed=42,
        max_tokens=8000,  # This max token works
        max_token_length=8000,  # This doesnt
    )
    return llm
