"""Readable console labels; raw stage keys and measured values stay unchanged."""
STAGES={
    'generation_start':'Starting model answers for the current question',
    'generation':'Writing model answers for the current question',
    'generation_finished':'Model answers generated; not an optimizer update',
    'generation_oom_retry':'Retrying the same answers after GPU memory exhaustion',
    'rollout':'Answers checked against the reward function',
    'gradient_scoring':'Question gradient score completed',
    'policy_update':'Optimizer update completed', 'update':'Training update completed',
    'model_ready':'Model loaded and ready', 'checkpoint_load':'Loading saved model and optimizer state',
    'checkpoint_save':'Saving model, optimizer and RNG state',
    'selection':'Scoring questions to choose the next training batch',
    'training':'Computing the next optimizer update', 'preparation':'Preparing training questions',
    'startup':'Preparing the model', 'cache_generation':'Generating answers to build the training reward cache',
    'cache_refresh':'Refreshing the training reward cache', 'cache_export':'Saving the training reward cache',
    'evaluation':'Measuring answers on the held-out evaluation questions',
}
COHORTS={'olmo_math':'OLMo / MATH','qwen_mbpp':'Qwen / MBPP'}


def describe(stage,details):
    parts=[STAGES.get(stage,'Recorded work: '+str(stage))]
    if details.get('cohort'):parts.insert(0,COHORTS.get(details['cohort'],details['cohort']))
    if details.get('purpose'):parts.append('purpose: '+details['purpose'])
    if details.get('questions_total') is not None:
        total=details['questions_total'];index=details.get('question_index')
        if index:parts.append(f'question {index}/{total} on this GPU')
        if details.get('questions_answered') is not None:parts.append(f"answers checked for {details['questions_answered']}/{total} questions")
        if details.get('gradient_scores_completed') is not None:parts.append(f"gradient scores {details['gradient_scores_completed']}/{total} complete")
    if details.get('responses_requested') is not None:parts.append(f"requesting {details['responses_requested']} answers")
    if details.get('response_tokens') is not None:
        text=f"current answer length {details['response_tokens']} tokens"
        if details.get('max_new_tokens') is not None:text+=f" (limit {details['max_new_tokens']})"
        parts.append(text)
    elif details.get('sequence_tokens') is not None:parts.append(f"sequence length {details['sequence_tokens']} tokens including input")
    if details.get('prompt') is not None:parts.append(f"question ID {details['prompt']}")
    if details.get('optimizer_steps_this_process') is not None:parts.append(f"optimizer updates completed in this process: {details['optimizer_steps_this_process']}")
    if details.get('step') is not None:parts.append(f"recorded training update {details['step']}")
    if details.get('progress_kind')=='entered_only':parts.append('no newly decoded token recorded yet')
    return ' | '.join(parts)
