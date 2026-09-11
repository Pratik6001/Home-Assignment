import pandas as pd
import kagglehub
import os

def main():
    print("Finding downloaded dataset path...")
    path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
    twcs_path = os.path.join(path, "twcs", "twcs.csv")
    
    if not os.path.exists(twcs_path):
        print(f"Error: Could not find twcs.csv at {twcs_path}")
        return

    print("Loading dataset (this may take a minute)...")
    df = pd.read_csv(twcs_path)
    
    # We want to pair each SpotifyCares response with the customer's initial message
    # 'in_response_to_tweet_id' in SpotifyCares response points to the customer's tweet
    
    # Cast to Int64 to remove .0 from floats before converting to str
    df['in_response_to_tweet_id_str'] = pd.to_numeric(df['in_response_to_tweet_id'], errors='coerce').astype('Int64').astype(str)
    df['tweet_id_str'] = df['tweet_id'].astype(str)
    
    spotify_responses = df[df['author_id'] == 'SpotifyCares'].copy()
    print(f"Found {len(spotify_responses)} responses from @SpotifyCares")
    
    customer_tweet_ids = spotify_responses['in_response_to_tweet_id_str'].tolist()
    
    # Get the customer tweets
    customer_tweets = df[df['tweet_id_str'].isin(customer_tweet_ids)].copy()
    
    # Merge to create pairs
    merged = pd.merge(
        customer_tweets, 
        spotify_responses, 
        left_on='tweet_id_str', 
        right_on='in_response_to_tweet_id_str', 
        suffixes=('_customer', '_brand')
    )
    
    print(f"Formed {len(merged)} valid Customer-Brand conversation pairs.")
    
    # Select and rename columns for a cleaner dataset
    final_df = merged[[
        'tweet_id_customer',
        'text_customer',
        'created_at_customer',
        'tweet_id_brand',
        'text_brand',
        'created_at_brand'
    ]].copy()
    
    # Sample down to 5000 for our dev/training set to keep it lightweight
    if len(final_df) > 5000:
        dev_set = final_df.sample(n=5000, random_state=42)
    else:
        dev_set = final_df
        
    os.makedirs('data', exist_ok=True)
    dev_set.to_csv('data/spotify_threads.csv', index=False)
    print("Saved 5000 threads to data/spotify_threads.csv")
    
    # Sample 200 for the golden eval set (distinct from the 5000 dev set if possible)
    # We'll just take 200 from the remainder
    remaining = final_df.drop(dev_set.index)
    if len(remaining) >= 200:
        golden_set = remaining.sample(n=200, random_state=42)
    else:
        golden_set = final_df.sample(n=200, random_state=42)
        
    # We will save the golden set without the brand's response text in the 'input' side, 
    # but we'll keep the actual response as 'reference_reply'
    golden_df = golden_set[['tweet_id_customer', 'text_customer', 'text_brand']].copy()
    golden_df.rename(columns={
        'tweet_id_customer': 'id',
        'text_customer': 'customer_message',
        'text_brand': 'actual_brand_reply'
    }, inplace=True)
    
    # Add empty columns for our labelling later
    golden_df['intent'] = ''
    golden_df['ideal_escalate'] = ''
    
    golden_df.to_csv('data/golden_eval_unlabelled.csv', index=False)
    print("Saved 200 unlabelled examples to data/golden_eval_unlabelled.csv")

if __name__ == '__main__':
    main()
