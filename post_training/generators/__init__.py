"""Synthetic data generators for historical LLM post-training."""


def get_generator_registry():
    """Lazy import to avoid errors when individual generators don't exist yet."""
    from src.post_training.generators.gen_a_factual import GenAFactual
    from src.post_training.generators.gen_b_cot import GenBCoT
    from src.post_training.generators.gen_c_comprehension import GenCComprehension
    from src.post_training.generators.gen_d_temporal import GenDTemporal
    from src.post_training.generators.gen_e_quantitative import GenEQuantitative
    from src.post_training.generators.gen_f_completion import GenFCompletion
    from src.post_training.generators.gen_g_instruct import GenGInstruct

    return {
        "A": GenAFactual,
        "B": GenBCoT,
        "C": GenCComprehension,
        "D": GenDTemporal,
        "E": GenEQuantitative,
        "F": GenFCompletion,
        "G": GenGInstruct,
    }
