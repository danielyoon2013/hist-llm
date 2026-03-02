"""Synthetic data generators for historical LLM post-training."""


def get_generator_registry():
    """Lazy import to avoid errors when individual generators don't exist yet."""
    from src.post_training.generators.gen_a_factual import GenAFactual
    from src.post_training.generators.gen_b_cot import GenBCoT
    from src.post_training.generators.gen_c_comprehension import GenCComprehension
    from src.post_training.generators.gen_d_quantitative import GenDQuantitative
    from src.post_training.generators.gen_e_completion import GenECompletion
    from src.post_training.generators.gen_f_instruct import GenFInstruct

    return {
        "A": GenAFactual,
        "B": GenBCoT,
        "C": GenCComprehension,
        "D": GenDQuantitative,
        "E": GenECompletion,
        "F": GenFInstruct,
    }
