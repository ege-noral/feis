from .loader import load_subject_dataframe 
from helper import split_into_epochs

def get_epochs( 
    subject_id,
):

    resting_trials = get_resting_trial_epochs(subject_id)
    thinking_trials = get_thinking_trial_epochs(subject_id)

    return resting_trials, thinking_trials

def get_thinking_trial_epochs( 
    subject_id,
):

    thinking_df = load_subject_dataframe(
        subject_id=subject_id,
        condition="thinking"
    )
    thinking_trials = split_into_epochs(thinking_df)

    return thinking_trials
    
    
def get_resting_trial_epochs( 
    subject_id,
):

    resting_df = load_subject_dataframe(
        subject_id=subject_id,
        condition="resting"
    )
    resting_trials = split_into_epochs(resting_df)

    return resting_trials
    
   