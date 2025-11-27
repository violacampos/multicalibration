from tools import  binning, cmd_input
from tools.calibration_scores import Score



def compute_uncalibrated_scores(score_obj, data_provider, args):
    """Calculate all scores for the uncalibrated test set."""
    test_probs = data_provider.get_test_probs(args.prob_method)
    test_is_correct = data_provider.get_test_is_correct()
    test_groups = data_provider.get_test_groups()
    
    scores = score_obj.calc_all(
        test_probs,
        test_is_correct,
        groups=test_groups,
        set_brier_ref=True
    )
    
    return scores

def build_return_dict(data_provider, args, scores, total_bin, correctness_bin,
                      total_group, correctness_group, avg_group_confidence):
    """Build dictionary of results to return when called externally."""
    return {
        "correctness_bin_uncalibrated": correctness_bin,
        "total_bin_uncalibrated": total_bin,
        "correctness_group_uncalib": correctness_group,
        "average_group_confidence_uncalib": avg_group_confidence,
        "total_group_uncalib": total_group,
        "scores_uncalibrated": scores,
        "uncalibrated_probs": data_provider.get_test_probs(args.prob_method),
        "is_correct": data_provider.get_test_is_correct(),
        "groups": data_provider.get_test_groups(),
        "language": data_provider.get_test_languages(),
        #"names": data_provider.get_test_names(), TODO use ids instead
        "programs": data_provider.get_test_programs(),
        "prompts": data_provider.get_test_prompts(),
        "token_logprobs": data_provider.get_test_token_logprobs(),
        "group_names": data_provider.group_names
    }


def main(data_provider, bins=None):
    """
    Compute baseline (uncalibrated) scores for model predictions.
    
    Args:
        data_provider: Object providing access to test data

        grid: Optional pre-computed binning grid

    
    Returns:
        Dictionary of results 
    """
    args = cmd_input.load_parser()
    
    # Setup grid
    if bins is None:
        bins = binning.Binning(args.bin_count, args.binning_type)
    
    # Initialize scoring object
    score_obj = Score(bins, args)
    
    # Compute all metrics
    test_probs = data_provider.get_test_probs(args.prob_method)
    test_is_correct = data_provider.get_test_is_correct()
    test_groups = data_provider.get_test_groups()
    
    scores_uncalibrated = compute_uncalibrated_scores(score_obj, data_provider, args)
    # binning stats
    _, _, total_bin, correctness_bin = score_obj.get_total_and_correctness(
        test_probs,
        test_is_correct,
        test_groups
    )
    # group stats
    total_group, correctness_group, avg_group_confidence = score_obj.get_correctness_per_group(
        test_probs,
        test_is_correct,
        test_groups
    )
    

    return build_return_dict(
        data_provider, args, scores_uncalibrated,
        total_bin, correctness_bin,
        total_group, correctness_group, avg_group_confidence
    )

