import boto3
import os
import io
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import ftfy
   
paidcountry_mapping = {
    "Sell On Amazon":'IN',
    "Amazon AU - OGB":'AU',
    "FR - LUX Entity":'FR',
    "Amazon Australia LinkedIn":'AU',
    "DE - LUX Entity":'DE',
    "Amazon FR Corp PR 2022 - EUR":'FR',
    "Amazon GSMC APAC":'APAC',
    "AmazonNewsES - Ogilvy":'ES',
    'Amazon Japan [PR JP]':'JP',
    'ES - LUX Entity':'ES' ,
    'AU SMC PR FB 2P # Ad Account':'AU',
    'IT - LUX Entity':'IT',
    'UK - LUX Entity':'UK',
    'AmazonNews EU - Amazon':'EU',
    'AU WWC GSMC FB 2P # Consumer PR':'AU',
    'UK - UK Entity':'UK',
    'Amazon DE Corp PR 2022 (EUR 51 PO)':'DE',
    'Highlights - LUX Entity':'Global',
    'DE – DE Entity':'DE',
    'ES – ES Entity':'ES',
    'IT – IT Entity':'IT'

}


paidregion_mapping = {
    "Sell On Amazon":'APAC',
    "Amazon AU - OGB":'APAC',
    "FR - LUX Entity":'Europe',
    "Amazon Australia LinkedIn":'APAC',
    "DE - LUX Entity":'Europe',
    "Amazon FR Corp PR 2022 - EUR":'Europe',
    "Amazon GSMC APAC":'APAC',
    "AmazonNewsES - Ogilvy":'Europe',
    'Amazon Japan [PR JP]':'APAC' ,
    'ES - LUX Entity':'Europe' , 
    'AU SMC PR FB 2P # Ad Account':'APAC',
    'IT - LUX Entity':'Europe',
    'UK - LUX Entity':'Europe',
     'AmazonNews EU - Amazon':'Europe',
     'AU WWC GSMC FB 2P # Consumer PR':'APAC',
     'UK - UK Entity':'Europe',
     'Amazon DE Corp PR 2022 (EUR 51 PO)':'Europe',
     'Highlights - LUX Entity':'Global',
    'DE – DE Entity':'Europe',
    'ES – ES Entity':'Europe',
    'IT – IT Entity':'Europe'

}


    

      
    
platform = {
    'FBPAGE': 'FACEBOOK',
    'INSTAGRAM': 'INSTAGRAM',
    'LINKEDIN_COMPANY': 'LINKEDIN_COMPANY',
    'TWITTER': 'TWITTER',
    'TWITTER_AD_ACCOUNT': 'TWITTER',
    'YOUTUBE': 'YOUTUBE',
    'TIKTOK_BUSINESS':'TikTok'
     }
  
PaidDataColumns=['FACEBOOK_AVG__DURATION_OF_VIDEO_PLAYED__SUM',
       'FACEBOOK_COST_PER_DAILY_ESTIMATED_AD_RECALL_LIFT__PEOPLE__IN_DEFAULT__SUM',
       'COST_PER_1_000_IMPRESSIONS__CPM__IN_DEFAULT__SUM', 'IMPRESSIONS__SUM',
       'LINK_CLICKS__FB___LI___TW___SUM', 'ACM_GLOBAL_SHARES__SUM',
       'ANKUR_VIDEO_THRU_PLAY_OR_15_SEC_IN_DEFAULT__SUM',
       'DAILY_ESTIMATED_AD_RECALL_LIFT__PEOPLE___SUM', 'AD_ACCOUNT',
       'TOTAL_RESULTS_FOR_OBJECTIVE__SUM', 'AD_VARIANT_NAME',
       'ACM_GLOBAL_COMMENTS__SUM', 'CHANNEL', 'ACM_GLOBAL_LIKES__SUM',
       'PERMALINK', 'AD_OBJECTIVE', 'CLICKS__SUM',
       'ACM_GLOBAL_TOTAL_ENGAGEMENTS__SUM', 'ACM_GLOBAL_VIDEO_VIEWS__SUM',
       'SPENT__USD__IN_USD__SUM', 'ACM_GLOBAL_CONVERSIONS__SUM',
       'RESULTS_PER_AMOUNT_SPENT__SUM', 'Objective', 'Cleaned', 'Key Metric',
       'Key Metric 2', 'Key Metric 3','is_paiddata','Paid_Reach']
    
    


rep_mapping = {
'Engagement to drive Brand Love' :'Brand Building',
'Products and Services (Prime, Price, Selection, Convenience, Products)':'Products & Services',
'Workplace (Amazon as a good employer)':'Workplace',
'Community (charity and outreach)':'Community (charity and outreach)',
'Diversity Equity and inclusion':'DEI',
'Supporting Small Business':'Supporting SMB',
'Sustainability (all sustainability)':'Sustainability',
'Customer Trust (reviews, counterfeit)':'Customer Trust (reviews, counterfeit)'
}

content_mapping = {
'Type 1: Straight news amplification (i.e. sharing a blog or media coverage)':'Type 1: News and Announcements',
'Type 2: Original content to support company news (i.e. making an original video for a store or device launch)':'Type 2: PR Campaigns',
'Type 3: Original evergreen (i.e. repurposing employee-generated content for our own post)':'Type 3: Evergreen Content (GSMC)'
}

who_mapping = {
'Employee': 'Employee',
'Agency': 'PR',
'Creator/influencer':'Creator/Influencer',
'Customer':'Customer (UGC)',
'GSMC - EMEA':'GSMC-EMEA'
}

##For impressions if platform = instagram & impressions = 0 give reel plays otherwise give impressions
##Same for video views
  
      
impressioncols = [
'LINKEDIN_COMPANY_POST_IMPRESSIONS__SUM',
'INSTAGRAM_BUSINESS_POST_IMPRESSIONS__SUM',
'FACEBOOK_POST_PAID_IMPRESSIONS__SUM',
'FACEBOOK_POST_PAID_IMPRESSIONS',
'FACEBOOK_POST_ORGANIC_IMPRESSIONS',
'X_IMPRESSIONS__SUM',
'YOUTUBE_VIDEO_VIEWS__SUM',
'INSTAGRAM_BUSINESS_POST_REEL_PLAYS__SUM',
'IMPRESSIONS__SUM',"INSTAGRAM_BUSINESS_POST_TOTAL_REEL_PLAYS__SUM","TIKTOK_VIDEO_VIEWS__SUM","TIKTOK_VIDEO_VIEWS"
] 
#'IMPRESSIONS__SUM'->paid "INSTAGRAM_BUSINESS_POST_TOTAL_REEL_PLAYS__SUM",   INSTAGRAM_BUSINESS_POST_TOTAL_REEL_PLAYS__SUM" INSTAGRAM_BUSINESS_POST_TOTAL_REEL_PLAYS__SUM

paidimpcols = [
'FACEBOOK_POST_PAID_IMPRESSIONS__SUM','IMPRESSIONS__SUM','YOUTUBE_PAID_VIEWS__SUM','FACEBOOK_POST_PAID_IMPRESSIONS'
]
#'IMPRESSIONS__SUM'->paid

#checked POST_REACH__SUM to make sure it doesn't double count

reach = [
'INSTAGRAM_BUSINESS_POST_REACH__SUM',
'FACEBOOK_POST_ORGANIC_REACH',
'INSTAGRAM_STORY_REACH__SUM',
'YOUTUBE_VIDEO_VIEWS__SUM',
'POST_REACH__SUM',
'TIKTOK_VIDEO_REACH__SUM',
'TIKTOK_VIDEO_REACH'
]
 
##Instagram Business Post Saved (SUM), Instagram Post Likes (SUM), Instagram Post Comments (SUM)

##Youtube= likes, comments, shares


engagementcols = [
'LINKEDIN_COMPANY_POST_ORGANIC_ENGAGEMENTS__SUM',
'FACEBOOK_POST_ENGAGED_USERS',
'X_TOTAL_ENGAGEMENTS__SUM',
'INSTAGRAM_POST_COMMENTS__SUM',
'INSTAGRAM_BUSINESS_POST_SAVED__SUM',
'INSTAGRAM_POST_LIKES__SUM',
'YOUTUBE_VIDEO_COMMENTS__SUM',
'YOUTUBE_VIDEO_LIKES__SUM',
'YOUTUBE_VIDEO_SHARES__SUM',

'TIKTOK_VIDEO_COMMENTS__SUM',
'TIKTOK_VIDEO_LIKES__SUM',
'TIKTOK_VIDEO_SHARES__SUM',

'TIKTOK_VIDEO_COMMENTS',
'TIKTOK_VIDEO_LIKES',
'TIKTOK_VIDEO_SHARES',
'INSTAGRAM_BUSINESS_POST_SHARES__SUM'
]  
 
#'POST_SHARES__SUM','POST_SHARES',
     
#'INSTAGRAM_BUSINESS_POST_ENGAGEMENT__SUM',
 
likescols = [
'LINKEDIN_COMPANY_POST_LIKES_AND_REACTIONS__SUM',
'YOUTUBE_VIDEO_LIKES__SUM',
'POST_LIKES_AND_REACTIONS',
'TIKTOK_VIDEO_LIKES__SUM',
'POST_LIKES_AND_REACTIONS__SUM',
'TIKTOK_VIDEO_LIKES',
]

sharescols = [
'LINKEDIN_COMPANY_POST_SHARES__SUM',
'YOUTUBE_VIDEO_SHARES__SUM',
'INSTAGRAM_BUSINESS_POST_SHARES__SUM',
'TIKTOK_VIDEO_SHARES__SUM',
'TIKTOK_VIDEO_SHARES',
'POST_SHARES'
]
    
commentscols = [
'LINKEDIN_COMPANY_POST_COMMENTS__SUM',
'INSTAGRAM_POST_COMMENTS__SUM',
'YOUTUBE_VIDEO_COMMENTS__SUM',
'POST_COMMENTS__SUM',
'TIKTOK_VIDEO_COMMENTS__SUM',
'POST_COMMENTS',
'TIKTOK_VIDEO_COMMENTS'
]

 
videoviewscols = [
'LINKEDIN_VIDEO_VIEWS__SUM',
'INSTAGRAM_VIDEO_VIEWS__SUM',
'FACEBOOK_VIDEO_ORGANIC_VIEWS__VIEWED_FOR_3_SECONDS_OR_MORE',
'X_VIDEO_VIEWS__SUM',
'YOUTUBE_VIDEO_VIEWS__SUM',
'INSTAGRAM_BUSINESS_POST_REEL_PLAYS__SUM',
'TIKTOK_VIDEO_VIEWS__SUM',
'TIKTOK_VIDEO_VIEWS'
]

Organic_Video_Views_to_95 =[
'FACEBOOK_VIDEO_ORGANIC_VIEWS__VIEWED_95__TO_VIDEO_LENGTH___SUM',
'YOUTUBE_VIEW_GREATER_THAT_95___SUM',
'X_VIDEO_VIEWED_100___SUM',
'TIKTOK_VIDEO_WATCHED_TO_COMPLETION_RATE_IN_',
'FACEBOOK_VIDEO_ORGANIC_VIEWS__VIEWED_95__TO_VIDEO_LENGTH',
'TIKTOK_VIDEO_WATCHED_TO_COMPLETION_RATE'
    ]
     
thruplaycols = [
'ANKUR_VIDEO_THRU_PLAY_OR_15_SEC_IN_DEFAULT__SUM'
]


EXCLUDED_EXTENSIONS = [
'.png',
'.jpg',
'.mp4',
'.jpeg'
]


keep_permalinks = [
    'https://www.instagram.com/reel/C4QyFDIrqr3/'
    # Add more PERMALINKs to this list as needed
]

# Columns to be processed
columns_to_process = {
    'TWITTER_TOTAL_ENGAGEMENTS__SUM': 'X_TOTAL_ENGAGEMENTS__SUM',
    'TWITTER_VIDEO_VIEWED_95___DEPRECATED___SUM': 'X_VIDEO_VIEWED_100___SUM',
    'TWITTER_VIDEO_VIEWS__SUM': 'X_VIDEO_VIEWS__SUM',
    'TWITTER_IMPRESSIONS__SUM': 'X_IMPRESSIONS__SUM'
}

video_length=[
    'FACEBOOK_VIDEO_LENGTH__SUM',
    'TIKTOK_VIDEO_DURATION__SUM',
    'FACEBOOK_VIDEO_LENGTH',
    'TIKTOK_VIDEO_DURATION'
]

source_bucket = 'wikitablescrapexample'
source_prefix = 'amazon_sprinklr_pull/result/'
destination_bucket = 'wikitablescrapexample'
destination_prefix = 'amazon_sprinklr_pull/finalmaster/'
s3 = boto3.client('s3')
def upload_csv_to_s3(df, bucket, key):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue(), ContentType='text/csv')
def read_csv_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(BytesIO(obj['Body'].read()),low_memory=False)
    return df
def read_excel_from_s3(source_bucket, key):
    obj = s3.get_object(Bucket=source_bucket, Key=key)
    file_content=obj['Body'].read()
    read_excel_data=io.BytesIO(file_content)
    df = pd.read_excel(read_excel_data)
    return df

def load_files():
    source_bucket = 'wikitablescrapexample'
    folder = 'amazon_sprinklr_pull/mappingandbenchmark/'
    
    country_mapping = read_excel_from_s3(source_bucket, folder + 'countrymapping.xlsx')
    platform_mapping = read_excel_from_s3(source_bucket, folder + 'platformmapping.xlsx')
    twitter_mapping =  read_excel_from_s3(source_bucket, folder + 'twitter_mapping.xlsx')
    # benchmarks_median = read_excel_from_s3(source_bucket, folder + 'median_benchmarks.xlsx')
    # benchmarks_mean = read_excel_from_s3(source_bucket, folder + 'mean_benchmarks.xlsx')
    
    return country_mapping, platform_mapping, twitter_mapping
    


 
def delivery_type(row):
    if row['is_paiddata'] == 1:
        return 'Paid'
    else:
        if row['Paid Impressions'] > 0:
            return 'Boosted'
        else:
            return 'Organic'
        
            
def rename_or_fillna(df, old_col, new_col):
    # If both columns exist
    if old_col in df.columns and new_col in df.columns:
        # Fill the null values of new_col with values from old_col
        df[new_col] = df[new_col].fillna(df[old_col])
        # Optionally drop the old_col
        df.drop(old_col, axis=1, inplace=True)
    elif old_col in df.columns:
        # Rename old_col to new_col
        df.rename(columns={old_col: new_col}, inplace=True)
    return df
    

def clean_data(df, country_mapping, platform_mapping, twitter_mapping):
  
# Define a mask to identify rows where 'PERMALINK' is NaN, 'INSTAGRAM_STORY_REACH__SUM' > 0, and 'ACCOUNT_TYPE' is 'INSTAGRAM'
    mask = (df['PERMALINK'].isna()) & (df['INSTAGRAM_STORY_REACH__SUM'] > 0) & (df['ACCOUNT_TYPE'].str.lower() == 'instagram')
    
    # Use the mask to assign new values to the 'PERMALINK' column based on values in 'PUBLISHEDTIME' and 'ACCOUNT' columns
    df.loc[mask, 'PERMALINK'] = df.loc[mask, 'PUBLISHEDTIME'].astype(str) + df.loc[mask, 'ACCOUNT']

    # Set the 'ACCOUNT_TYPE' to 'instagram story' for rows matching the mask
    df.loc[mask, 'ACCOUNT_TYPE'] = 'instagram story'

   
    
    mask_instagram = df['ACCOUNT_TYPE'].str.lower() == 'instagram'
    df.loc[mask_instagram & df['PERMALINK'].str.contains('instagram.com/reel/'), 'ACCOUNT_TYPE'] = 'reels'
    df.loc[mask_instagram & df['PERMALINK'].str.contains('instagram.com/stories/'), 'ACCOUNT_TYPE'] = 'instagram story'
    df.loc[mask_instagram & df['PERMALINK'].str.contains('instagram.com/p/'), 'ACCOUNT_TYPE'] = 'instagram'
    
    twitter_mapping_dict = dict(zip(twitter_mapping['ACCOUNT'].str.lower(), twitter_mapping['ACCOUNT_TYPE'].str.lower()))
    
    mask = (df['ACCOUNT_TYPE'].str.lower() == 'twitter') & df['ACCOUNT'].str.lower().isin(twitter_mapping_dict.keys())
    df.loc[mask, 'ACCOUNT_TYPE'] = df.loc[mask, 'ACCOUNT'].str.lower().map(twitter_mapping_dict)

      
    df3 = pd.DataFrame()
    df2 = df.copy()
    
    df3 = df2[PaidDataColumns]
    df3['Account'] = df2['ACCOUNT'].str.lower()
    df3['Account Type'] = df2['ACCOUNT_TYPE'].str.lower()
    
 
    platform_mapping['Account Type'] = platform_mapping['Account Type'].str.lower()
    

        # Merging with platform_mapping and country_mapping
    df3 = df3.merge(platform_mapping, on='Account Type', how='left')
    
 
    
    # Merging with platform_mapping and country_mapping
 
    country_mapping['Account']=country_mapping['Account'].str.lower()
    df3 = df3.merge(country_mapping, on='Account', how='left')

     
    df3['row_number']=df2['row_number']
 

    # Create a mask to identify rows where 'is_paiddata' is 1
    mask_paid = df3['is_paiddata'] == 1
    
    # Remove leading and trailing whitespace from 'AD_ACCOUNT' column
    df3['AD_ACCOUNT'] = df3['AD_ACCOUNT'].str.strip()
    
    # For rows where 'is_paiddata' is 1, use the 'AD_ACCOUNT' column to map and assign 'Country' and 'Region'
    df3.loc[mask_paid, 'Country'] = df3.loc[mask_paid, 'AD_ACCOUNT'].map(paidcountry_mapping)
    df3.loc[mask_paid, 'Region'] = df3.loc[mask_paid, 'AD_ACCOUNT'].map(paidregion_mapping)
    
    # For rows where 'is_paiddata' is 1, assign the value in the 'CHANNEL' column to the 'Platform' column
    df3.loc[mask_paid, 'Platform'] = df3.loc[mask_paid, 'CHANNEL']
    
    # Replace 'Platform' values that are 'X' with 'Twitter'
    df3.loc[(mask_paid) & (df3['Platform'] == 'X'), 'Platform'] = 'Twitter'
    # Creating new columns based on transformations

       
    #df3['Video Length (SUM)'] = df2['FACEBOOK_VIDEO_LENGTH__SUM']
    df3['Video Length (SUM)'] = df2[video_length].sum(axis=1)
    
 
    #df3['Facebook Video Average Time Viewed (SUM)'] = df2['FACEBOOK_VIDEO_AVERAGE_TIME_VIEWED__SUM']
    df3['Facebook Video Average Time Viewed (SUM)'] = df2['FACEBOOK_VIDEO_AVERAGE_TIME_VIEWED'].combine_first(df2['FACEBOOK_VIDEO_AVERAGE_TIME_VIEWED__SUM'])

    #df3['PERMALINK'] = df2['PERMALINK']
    df3['Post Format'] = df2['MEDIA_TYPE']
    df3['Post'] = df2['OUTBOUND_POST'].apply(lambda x: ftfy.fix_text(x) if pd.notnull(x) else x)
    df3.loc[df3['Platform'].str.lower() == 'instagram story', 'Post'] = 'Instagram Story'
 
    df3['Published Date'] = df2['PUBLISHEDTIME']
     
 
    df3['Pull Date']=df2['Pull Date']
    
        

    

  

    df3['Reputational Topic'] = df2['GSMC__REPUTATIONAL_TOPIC__OUTBOUND_MESSAGE']
    df3['Content Category Type'] = df2['GSMC__CONTENT_CATEGORY_TYPE__OUTBOUND_MESSAGE']
    df3['Is it XGC?'] = df2['C_63DAC6AC6DF27A45C687B642']
    df3['If yes who made it?'] = df2['C_63DAC8FF6DF27A45C68AAF7C']
    df3['Tier 1 Event?'] = df2['GCCI_SOME__TIER_1_EVENT_TYPE__OUTBOUND_MESSAGE']
    df3['Post Message'] = df2['CAMPAIGN_NAME']
    df3['MESSAGE_TYPE']=df2['MESSAGE_TYPE']
    df3['IS_DARK_POST']=df2['IS_DARK_POST']
    

   
     
    # Updating 'MESSAGE_TYPE' column in df3 based on condition
    condition = df3['Platform'].isin(['YouTube', 'Instagram Story','TikTok'])
    df3.loc[condition, 'MESSAGE_TYPE'] = 'Update'

    # Replace mapped values
    df3['Reputational Topic'] = df3['Reputational Topic'].replace(rep_mapping)
    df3['Content Category Type'] = df3['Content Category Type'].replace(content_mapping)
    df3['If yes who made it?'] = df3['If yes who made it?'].replace(who_mapping)

    
      


    
    
    return df3



    
    
 
def calculate_metrics(df3,df):
    
    
    df2=df
   
    # Aggregating metrics from various columns
    df3['Total Impressions'] = df2[impressioncols].sum(axis=1)
     
     
    # print(set(df2['FACEBOOK_POST_PAID_IMPRESSIONS__SUM'][df2['ACCOUNT_TYPE'] == 'INSTAGRAM' ]))
         
 
     
    # return
    
    df3['Paid Impressions'] = df2[paidimpcols].sum(axis=1)
    df3['Organic Impressions'] = df3['Total Impressions'] - df3['Paid Impressions']
    df3.loc[df3['Organic Impressions'] < 0, 'Organic Impressions'] = 0
    df3['Organic Reach'] = df2[reach].sum(axis=1)

     
    
    # Calculating delivery type based on paid impressions
    
    df3['Delivery'] = df3.apply(delivery_type, axis=1)

    # Calculating various engagements and rates
    df3['Engagements'] = df2[engagementcols].sum(axis=1)
    
   
    #df3['Engagement Rate'] = df3['Engagements'] / df3['Organic Impressions']
    df3['Engagement Rate'] = df3['Engagements'] / df3['Total Impressions']
    df3['Likes'] = df2[likescols].sum(axis=1)
    df3['Likes per Impression'] = df3['Likes'] / df3['Total Impressions']
    df3['Shares'] = df2[sharescols].sum(axis=1)
    df3['Comments'] = df2[commentscols].sum(axis=1)
    df3['Comments per Impression'] = df3['Comments'] / df3['Total Impressions']
    df3['Video Views'] = df2[videoviewscols].sum(axis=1)
    df3['Organic_Video_Views_to_95%'] = df2[Organic_Video_Views_to_95].sum(axis=1)


# Adjusting the values in 'Organic_Video_Views_to_95%' based on 'Platform', taking nulls into account
    df3['Organic_Video_Views_to_95%'] = df3.apply(
    lambda row: (row['Organic_Video_Views_to_95%'] / 100 if not pd.isnull(row['Organic_Video_Views_to_95%']) else np.nan)
    if row['Platform'] == 'TikTok' else row['Organic_Video_Views_to_95%'], axis=1)
    df3['Completion Rate'] = df3['Organic_Video_Views_to_95%'] / df3['Video Views']
    
    df3.loc[(df3['Platform'] == 'YouTube') & (df3['Organic Impressions'] >= 0), 'Organic Reach'] = df3['Organic Impressions']
    df3.loc[(df3['Platform'] == 'YouTube') & (df3['Organic Impressions'] >= 0), 'Video Views'] = df3['Organic Impressions']
    
    condition = (df3['Platform'] == 'Facebook') & (df3['Post Format'] == 'Reel')


    df3.loc[condition, 'Engagements'] = df3.loc[condition, 'Likes'] + df3.loc[condition, 'Shares'] + df3.loc[condition, 'Comments']


    df3.loc[condition, 'Engagement Rate'] = df3.loc[condition, 'Engagements'] / df3.loc[condition, 'Total Impressions']
    condition1 = df3['Platform'] == 'Instagram Reels'
    
    
 
    df3.loc[condition1, 'Video Views'] = df3.loc[condition1, 'Organic Impressions']
  

    #df3['Comment Sentiment'] = df2['NET_SENTIMENT_IN_'] / 100
    
    combined_sentiment = df2['NET_SENTIMENT_IN_'].combine_first(df2['NET_SENTIMENT'])

# Divide by 100 to get the 'Comment Sentiment'
    df3['Comment Sentiment'] = combined_sentiment / 100
    
 
    
    
    
    columns_to_replace = ['Likes', 'Likes per Impression', 'Shares', 'Comments', 'Comments per Impression', 'Video Views', 'Organic_Video_Views_to_95%', 'Completion Rate', 'Comment Sentiment', 'Engagements', 'Engagement Rate']

    for col in columns_to_replace:
        df3.loc[df3['Platform'] == 'Instagram Story', col] = df3.loc[df3['Platform'] == 'Instagram Story', col].replace(0, np.nan)
        



       # # Filtering based on certain conditions
    df3 = df3[(df3['Total Impressions'] > 0) | (df3['is_paiddata'] == 1)]


    return df3
    
     
def filter_latest_pull_dates(df):
    # Reset the index for the main DataFrame
    df = df.reset_index(drop=True)
  
    # Define platform types
    organic_platform = ['Twitter Consumer', 'Instagram', 'Instagram Reels', 'Instagram Story', 'Twitter News', 'LinkedIn','TikTok']
    mixed_platform = ['Facebook', 'YouTube']
    # For organic platforms
    max_dates_dict = df[df['Platform'].isin(organic_platform) & (df['Delivery'] == 'Organic')].groupby('PERMALINK')['Pull Date'].max().to_dict()
    df['Max Pull Date'] = df['PERMALINK'].map(max_dates_dict)

    organic_latest_rows = df[(df['Platform'].isin(organic_platform)) & (df['Pull Date'] == df['Max Pull Date'])]

    organic_latest_rows = organic_latest_rows.groupby('PERMALINK').apply(lambda group: group.iloc[0]).reset_index(drop=True)

    # For mixed platforms
    # Calculate minimum pull date for each permalink
    min_pull_dates = df[df['Platform'].isin(mixed_platform)].groupby('PERMALINK')['Pull Date'].min()

    # Calculate maximum organic pull date for each permalink
    max_organic_pull_dates = df[(df['Platform'].isin(mixed_platform)) & (df['Delivery'] == 'Organic')].groupby('PERMALINK')['Pull Date'].max()

    # Calculate the differences
    differences = max_organic_pull_dates - min_pull_dates

    # Filter permalinks where the difference is greater than 5 days (organic for more than 5 days)
    organic_more_than_5_days_permalinks = differences[differences > pd.Timedelta(days=5)].index

    # Filter rows for permalinks that are organic for more than 5 days
    organic_more_than_5_days_rows = df[(df['Platform'].isin(mixed_platform)) & (df['Delivery'] == 'Organic') & (df['PERMALINK'].isin(organic_more_than_5_days_permalinks))]

    # Filter permalinks where the difference is less than or equal to 5 days (organic for less than or equal to 5 days)
    organic_less_than_5_days_permalinks = differences[(differences <= pd.Timedelta(days=5))].index

    # Filter rows for permalinks that are organic for less than or equal to 5 days
    organic_less_than_5_days_rows = df[(df['Platform'].isin(mixed_platform)) & (df['Delivery'] == 'Organic') & (df['PERMALINK'].isin(organic_less_than_5_days_permalinks))]

    # Identify permalinks that exist in boosted but not in organic_more_than_5_days_permalinks
    boosted_exclusive_permalinks = df[(df['Platform'].isin(mixed_platform)) & (df['Delivery'] == 'Boosted') & ~df['PERMALINK'].isin(organic_more_than_5_days_permalinks)]['PERMALINK'].unique()

    # Filter rows for permalinks that are boosted and not in organic_more_than_5_days_permalinks
    boosted_exclusive_rows = df[(df['Platform'].isin(mixed_platform)) & (df['Delivery'] == 'Boosted') & (df['PERMALINK'].isin(boosted_exclusive_permalinks))]

    # Remove rows from organic_less_than_5_days_rows that are also in boosted_exclusive_rows
    organic_less_than_5_days_rows = organic_less_than_5_days_rows[~organic_less_than_5_days_rows['PERMALINK'].isin(boosted_exclusive_rows['PERMALINK'])]

    # Combine the results
    result = pd.concat([organic_latest_rows,organic_more_than_5_days_rows, organic_less_than_5_days_rows, boosted_exclusive_rows], ignore_index=True)

    # Drop duplicates based on 'PERMALINK' while keeping the row with the latest 'Pull Date'
    result = result.sort_values('Pull Date').drop_duplicates('PERMALINK', keep='last').reset_index(drop=True)
    result.drop(columns=['Max Pull Date'], inplace=True, errors='ignore')


    return result
  
 
def filter_and_combine_paid_unpaid_data(df):
    
 
    # Separate the data based on is_paiddata
    unpaid_data = df[df['is_paiddata'] == 0]
    paid_data =df[df['is_paiddata'] == 1]

    
    common_permalinks = set(paid_data['PERMALINK']).intersection(set(unpaid_data['PERMALINK']))
 
    paid_data['updated_flag'] = 0
 

    columns_to_update = ['Account Type', 'Account', 'Platform', 'Published Date', 'Post Format', 'Post']
    for permalink in common_permalinks:
        # Make sure there is at least one unpaid row with the current permalink
        unpaid_rows_with_permalink = unpaid_data[unpaid_data['PERMALINK'] == permalink]
        if not unpaid_rows_with_permalink.empty:
            # Extract the first unpaid row (assuming you want to use the first one found)
            unpaid_row_to_use = unpaid_rows_with_permalink.iloc[0]
            for col in columns_to_update:
                # Update all matching rows in paid_data
                paid_data.loc[paid_data['PERMALINK'] == permalink, col] = unpaid_row_to_use[col]
            # Set updated_flag to 1 for all matching rows in paid_data
            paid_data.loc[paid_data['PERMALINK'] == permalink, 'updated_flag'] = 1

    # Apply the filter function to the unpaid data
    filtered_unpaid_data = filter_latest_pull_dates(unpaid_data)


    
    # Concatenate the filtered unpaid data with the paid data
    result = pd.concat([filtered_unpaid_data, paid_data], ignore_index=True)
    
    return result


def aggregate_paid_data(df):
    # Step 1: Create a new column for weighted average calculation later
    df['WA_FB_AVG__DURATION_OF_VIDEO_PLAYED__SUM'] = df['FACEBOOK_AVG__DURATION_OF_VIDEO_PLAYED__SUM'] * df['ACM_GLOBAL_VIDEO_VIEWS__SUM']



    aggregations = {
        'WA_FB_AVG__DURATION_OF_VIDEO_PLAYED__SUM':'sum',
        'FACEBOOK_COST_PER_DAILY_ESTIMATED_AD_RECALL_LIFT__PEOPLE__IN_DEFAULT__SUM': 'sum',
        'IMPRESSIONS__SUM': 'sum',
        'ACM_GLOBAL_SHARES__SUM': 'sum',
        'ACM_GLOBAL_LIKES__SUM': 'sum',
        'ANKUR_VIDEO_THRU_PLAY_OR_15_SEC_IN_DEFAULT__SUM': 'sum',
        'CLICKS__SUM': 'sum',
        'ACM_GLOBAL_TOTAL_ENGAGEMENTS__SUM': 'sum',
        'ACM_GLOBAL_VIDEO_VIEWS__SUM': 'sum',
        'SPENT__USD__IN_USD__SUM': 'sum',
        'ACM_GLOBAL_CONVERSIONS__SUM': 'sum',
        'RESULTS_PER_AMOUNT_SPENT__SUM': 'sum',
        'LINK_CLICKS__FB___LI___TW___SUM': 'sum',
        'TOTAL_RESULTS_FOR_OBJECTIVE__SUM': 'sum',
        'Total Impressions': 'sum',
        'DAILY_ESTIMATED_AD_RECALL_LIFT__PEOPLE___SUM': 'sum',
        'ACM_GLOBAL_COMMENTS__SUM': 'sum',
        'Paid_Reach':'sum',
        'AD_ACCOUNT': 'first',
        'AD_OBJECTIVE': 'first',
        'Objective': 'first',
        'Cleaned': 'first',
        'Key Metric': 'first',
        'Key Metric 2': 'first',
        'Key Metric 3': 'first',
        'Account Type': 'first',
        'Account': 'first',
        'Platform': 'first',
        'Published Date': 'first',
        'Post Format': 'first',
        'Post': 'first',
        'is_paiddata': 'first',
        'Region': 'first',
        'Country':'first',
        'Pull Date': 'first',
        'Delivery':'first' ,
        'row_number':'first'
     }

 
#  Reach & paid Video views->ACM_GLOBAL_VIDEO_VIEWS__SUM,'Total Engagements', 
    # Filter the dataframe for paid and unpaid data
    paid_data = df[df['is_paiddata'] == 1]
    unpaid_data = df[df['is_paiddata'] == 0]

    # Group and aggregate the paid data
    aggregated_paid_data = paid_data.groupby(['AD_VARIANT_NAME', 'CHANNEL', 'PERMALINK','Objective'], as_index=False).agg(aggregations)
    
    # Calculate the overall average duration by dividing the sum of our new column by the sum of 'ACM_GLOBAL_VIDEO_VIEWS__SUM'
    aggregated_paid_data['FACEBOOK_AVG__DURATION_OF_VIDEO_PLAYED__SUM'] = aggregated_paid_data['WA_FB_AVG__DURATION_OF_VIDEO_PLAYED__SUM'] / aggregated_paid_data['ACM_GLOBAL_VIDEO_VIEWS__SUM']
    
    # Rename 'ACM_GLOBAL_VIDEO_VIEWS__SUM' to 'paid Video Views' and then drop the original column
    aggregated_paid_data.rename(columns={'ACM_GLOBAL_VIDEO_VIEWS__SUM': 'paid Video Views'}, inplace=True)
    aggregated_paid_data.rename(columns={'ANKUR_VIDEO_THRU_PLAY_OR_15_SEC_IN_DEFAULT__SUM': 'Thruplay'}, inplace=True)
    
    aggregated_paid_data['Total Impressions'] = aggregated_paid_data['IMPRESSIONS__SUM']
    
    


    # Concatenate aggregated paid data with unpaid data to get the final dataframe
    final_data = pd.concat([aggregated_paid_data, unpaid_data], ignore_index=True)

    # Optionally, drop the 'WA_FB_AVG__DURATION_OF_VIDEO_PLAYED__SUM' column as it is no longer needed
    final_data.drop(columns=['WA_FB_AVG__DURATION_OF_VIDEO_PLAYED__SUM','ANKUR_VIDEO_THRU_PLAY_OR_15_SEC_IN_DEFAULT__SUM','ACM_GLOBAL_VIDEO_VIEWS__SUM'], inplace=True)
    

    
    return final_data
    
    
def assign_new_columns(df):
    for index, row in df.iterrows():
        key_metric_1 = row['Key Metric']
        key_metric_2 = row['Key Metric 2']
        key_metric_3 = row['Key Metric 3']

        if pd.notna(key_metric_1):
            df.loc[index, key_metric_1] = 'Value for ' + key_metric_1

        if pd.notna(key_metric_2):
            df.loc[index, key_metric_2] = 'Value for ' + key_metric_2

        if pd.notna(key_metric_3):
            df.loc[index, key_metric_3] = 'Value for ' + key_metric_3

    return df
     
def calculate_metrics_for_paid(df):

    # Calculate CPE (Cost per Engagement)
    df['CPE'] = np.where(
        (pd.notna(df['SPENT__USD__IN_USD__SUM']) & pd.notna(df['ACM_GLOBAL_TOTAL_ENGAGEMENTS__SUM']) & (df['ACM_GLOBAL_TOTAL_ENGAGEMENTS__SUM'] != 0)),
        df['SPENT__USD__IN_USD__SUM'] / df['ACM_GLOBAL_TOTAL_ENGAGEMENTS__SUM'],
        np.nan
    )

    # Calculate CPC (Cost per Click)
    df['CPC'] = np.where(
        (pd.notna(df['SPENT__USD__IN_USD__SUM']) & pd.notna(df['CLICKS__SUM']) & (df['CLICKS__SUM'] != 0)),
        df['SPENT__USD__IN_USD__SUM'] / df['CLICKS__SUM'],
        np.nan
    )

    # Calculate CPM (Cost per Mille/Thousand Impressions)
    df['CPM'] = np.where(
        (pd.notna(df['SPENT__USD__IN_USD__SUM']) & pd.notna(df['IMPRESSIONS__SUM']) & (df['IMPRESSIONS__SUM'] != 0)),
        (df['SPENT__USD__IN_USD__SUM'] / df['IMPRESSIONS__SUM']) * 1000,
        np.nan
    )

    # Calculate Cost per Thruplay
    df['Cost per Thruplay'] = np.where(
        (pd.notna(df['SPENT__USD__IN_USD__SUM']) & pd.notna(df['Thruplay']) & (df['Thruplay'] != 0)),
        df['SPENT__USD__IN_USD__SUM'] / df['Thruplay'],
        np.nan
    )
  
    # Calculate Estimated Ad Recall Lift Rate
    df['Estimated Ad Recall Lift Rate'] = np.where(
        (pd.notna(df['DAILY_ESTIMATED_AD_RECALL_LIFT__PEOPLE___SUM']) & pd.notna(df['IMPRESSIONS__SUM']) & (df['IMPRESSIONS__SUM'] != 0)),
        df['DAILY_ESTIMATED_AD_RECALL_LIFT__PEOPLE___SUM'] / df['IMPRESSIONS__SUM'],
        np.nan
    )

    # Calculate Cost per Estimated Ad Recall Lift
    df['Cost per Estimated Ad Recall Lift'] = np.where(
        (pd.notna(df['SPENT__USD__IN_USD__SUM']) & pd.notna(df['DAILY_ESTIMATED_AD_RECALL_LIFT__PEOPLE___SUM']) & (df['DAILY_ESTIMATED_AD_RECALL_LIFT__PEOPLE___SUM'] != 0)),
        df['SPENT__USD__IN_USD__SUM'] / df['DAILY_ESTIMATED_AD_RECALL_LIFT__PEOPLE___SUM'],
        np.nan
    )

    # Record Total Engagements
    df['Total Engagements'] = np.where(
        pd.notna(df['ACM_GLOBAL_TOTAL_ENGAGEMENTS__SUM']),
        df['ACM_GLOBAL_TOTAL_ENGAGEMENTS__SUM'],
        np.nan
    )
 
    # Calculate CTR (Click Through Rate)
    df['CTR'] = np.where(
        (pd.notna(df['CLICKS__SUM']) & pd.notna(df['IMPRESSIONS__SUM']) & (df['CLICKS__SUM'] != 0)),
        (df['CLICKS__SUM'] / df['IMPRESSIONS__SUM']) * 100,
        np.nan
    )

    # Record Video Average Play Time
    df['Video Average Play Time'] = np.where(
        (pd.notna(df['FACEBOOK_AVG__DURATION_OF_VIDEO_PLAYED__SUM']) & (df['FACEBOOK_AVG__DURATION_OF_VIDEO_PLAYED__SUM'] != 0)),
        df['FACEBOOK_AVG__DURATION_OF_VIDEO_PLAYED__SUM'],
        np.nan
    )
  
    df['CPV'] = np.where((df['SPENT__USD__IN_USD__SUM'] != 0) & (df['paid Video Views'] != 0),
                              df['SPENT__USD__IN_USD__SUM'] / df['paid Video Views'],
                              np.nan)
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df

   

def process_paid_data_for_metrics_calculation(df):
    # Dividing the data into paid and unpaid
    paid_data = df[df['is_paiddata'] == 1]
    unpaid_data = df[df['is_paiddata'] == 0]

    # Applying both functions sequentially on the paid data DataFrame
    paid_data = assign_new_columns(paid_data.copy())
    paid_data = calculate_metrics_for_paid(paid_data)
    
    # Combining paid and unpaid data
    combined_data = pd.concat([paid_data, unpaid_data], ignore_index=True)
    
    return combined_data
    
    
    

def transfer_paid_to_boosted_on_matching(dataframe):
    boosted_data = dataframe[dataframe['Delivery'] == 'Boosted'].copy()
    paid_data = dataframe[dataframe['Delivery'] == 'Paid'].copy()
    organic_data = dataframe[dataframe['Delivery'] == 'Organic'].copy()
    
    # Initializing 'boostedandpaid' column to 0
    boosted_data['boostedandpaid'] = 0
    paid_data['boostedandpaid'] = 0
    organic_data['boostedandpaid'] = 0

    boosted_permalink_set = set(boosted_data['PERMALINK'].values)
    paid_permalink_set = set(paid_data['PERMALINK'].values)
    matching_permalink_set = boosted_permalink_set.intersection(paid_permalink_set)

    for permalink in matching_permalink_set:
        boosted_index = boosted_data[boosted_data['PERMALINK'] == permalink].index[0]
        paid_index = paid_data[paid_data['PERMALINK'] == permalink].index[0]
        
        # Setting 'boostedandpaid' flag to 1 for matched rows
        boosted_data.at[boosted_index, 'boostedandpaid'] = 1

        # Transferring 'impression_sum' from 'Paid' to 'Boosted'
        boosted_data.at[boosted_index, 'IMPRESSIONS__SUM'] = paid_data.at[paid_index, 'IMPRESSIONS__SUM']
        paid_data.at[paid_index, 'IMPRESSIONS__SUM'] = None
        
        # Updating 'total_impression' in 'Boosted' to be equal to 'impression_sum'
        boosted_data.at[boosted_index, 'Total Impressions'] = boosted_data.at[boosted_index, 'IMPRESSIONS__SUM']
        

        # After updating 'Total Impressions'
        boosted_data.at[boosted_index, 'Engagement Rate'] = boosted_data.at[boosted_index, 'Engagements'] / boosted_data.at[boosted_index, 'Total Impressions']


        # Transferring 'Thruplay' from 'Paid' to 'Boosted'
        boosted_data.at[boosted_index, 'Thruplay'] = paid_data.at[paid_index, 'Thruplay']
        paid_data.at[paid_index, 'Thruplay'] = None
        boosted_data.at[boosted_index, 'Thruplay'] = boosted_data.at[boosted_index, 'Thruplay']
        
        # Transferring 'CLICKS__SUM' from 'Paid' to 'Boosted'
        boosted_data.at[boosted_index, 'CLICKS__SUM'] = paid_data.at[paid_index, 'CLICKS__SUM']
        paid_data.at[paid_index, 'CLICKS__SUM'] = None
        
        # Transferring 'paid Video Views from 'Paid' to 'Boosted'
        boosted_data.at[boosted_index, 'paid Video Views'] = paid_data.at[paid_index, 'paid Video Views']
        paid_data.at[paid_index, 'paid Video Views'] = None
        
        
        # Transferring 'Paid_Reach' from 'Paid' to 'Boosted'
        boosted_data.at[boosted_index, 'Paid_Reach'] = paid_data.at[paid_index, 'Paid_Reach']
        paid_data.at[paid_index, 'Paid_Reach'] = None
        
            

    final_data = pd.concat([boosted_data, paid_data, organic_data])
     
    return final_data 
#Paid_Reachm                          
                    
                               
def calculate_cpm(row):
    if (('CPM' in [row['Key Metric'], row['Key Metric 2'], row['Key Metric 3']]) and 
        row['Delivery'] == 'Paid' and 
        row['SPENT__USD__IN_USD__SUM'] > 0 and 
        row['Total Impressions'] > 0):
        return (row['SPENT__USD__IN_USD__SUM'] / row['Total Impressions']) * 1000
    else:
        return row['CPM']  # Return existing CPM value if conditions are not met
                                                                                                           
def lambda_handler(event, context): 
    # Define the bucket and CSV file to read
    key = 'amazon_sprinklr_pull/monthlybackeup/backupmaster_2024-05-20.csv' 
    #key=event['Records'][0]['s3']['object']['key']
                              
    # example string  
    # extract the dates   
    basename = os.path.basename(key)
    dateintervel = basename.split('_')[1:3]
    dateintervel = '_'.join(dateintervel)

    
    # Read the CSV file into a pandas DataFrame
    df = read_csv_from_s3(source_bucket, key)
    df.drop(columns=['Region'], inplace=True, errors='ignore')

   
    
    for old_col, new_col in columns_to_process.items():
        df = rename_or_fillna(df, old_col, new_col)
        
    df['SPENT__USD__IN_USD__SUM'] = df['SPENT__USD__IN_USD__SUM'].fillna(0) + df['TWITTER_POST_SPENT__USD__IN_USD__SUM'].fillna(0)
    df['IMPRESSIONS__SUM'] = df['IMPRESSIONS__SUM'].fillna(0) + df['TWITTER_POSTS_IMPRESSIONS__SUM'].fillna(0)
    df = df.drop(columns=['TWITTER_POSTS_IMPRESSIONS__SUM', 'TWITTER_POST_SPENT__USD__IN_USD__SUM'])

    
    country_mapping, platform_mapping,twitter_mapping = load_files()
    
    df = df.reset_index(drop=True)
    
    cleandf = clean_data(df, country_mapping, platform_mapping,twitter_mapping)
    
     
 
    
    cleandf = calculate_metrics(cleandf,df)


       


    
    # Define the conditions for filtering
    condition_is_paiddata = cleandf['is_paiddata'] == 1
    condition_message_type = cleandf['MESSAGE_TYPE'].isin(['Update', 'Twitter Sent Reply','X Sent Reply'])
    
    
    # Combine the conditions
    # Select all rows where is_paiddata is 1, or (is_paiddata is 0 and MESSAGE_TYPE is either 'Update' or 'Twitter Sent Reply')
    condition_combined = condition_is_paiddata | ((cleandf['is_paiddata'] == 0) & condition_message_type)
     
    # Apply the combined condition to filter the DataFrame and assign it back to df3
    cleandf = cleandf[condition_combined]

      
    cleandf = cleandf.reset_index(drop=True)
    
    cleandf['Pull Date'] = pd.to_datetime(cleandf['Pull Date'])
    
    

    
    
  
    
    final_df = filter_and_combine_paid_unpaid_data(cleandf)

      
    final_df['Post'] = final_df['Post'].fillna('')
     

    final_df = final_df[(final_df['Post'] != 'This message has no text.') | (final_df['PERMALINK'].isin(keep_permalinks) & (final_df['Post'] == 'This message has no text.'))]

    final_df = final_df[~final_df['Post'].str.lower().str.contains('|'.join(EXCLUDED_EXTENSIONS))]
    patterns_to_drop = ['updated their cover photo.', 'updated their phone number.']
    
    # Escape special characters (like ".") outside of the f-string
    patterns_to_drop_escaped = [pattern.replace('.', '\\.') for pattern in patterns_to_drop]
    
    # Combine the patterns into a single regex pattern
    # Prepend with '(?i)' for case-insensitive matching
    pattern = '(?i)' + '|'.join(patterns_to_drop_escaped)
    
    # Filter out rows where 'Post' column matches the patterns
    # Note: This assumes 'Post' column exists and is of string type
    final_df = final_df[~final_df['Post'].str.contains(pattern, regex=True, na=False)]

  
     
    final_df=final_df.reset_index(drop=True)

  
    final_df=aggregate_paid_data(final_df)
 

    final_df=process_paid_data_for_metrics_calculation(final_df)
    
    final_df=transfer_paid_to_boosted_on_matching(final_df)

 
 
    
            # Filtering the DataFrame df3 to remove specific rows
    final_df = final_df[~((final_df['Delivery'].isin(['Organic', 'Boosted'])) & (final_df['Total Impressions'] < 250))]
    
           
  
     
    #final_df = final_df.reset_index(drop=True)
    
    final_df['total video views'] = final_df['paid Video Views'].fillna(0) + final_df['Video Views'].fillna(0)
    condition1 = final_df['Platform'] == 'Instagram Reels'

    final_df.loc[condition1, 'total video views'] = final_df.loc[condition1, 'Total Impressions']

    
    final_df['Total reach'] =  final_df['Organic Reach'].fillna(0) + final_df['Paid_Reach'].fillna(0)
        # Apply the function to each row of the dataframe
    final_df['CPM'] = final_df.apply(calculate_cpm, axis=1)
      
    #     # Condition to exclude rows where 'IS_DARK_POST' is 'TRUE', and not to exclude when it's 'FALSE', null, or an empty string
    # # under the specific conditions of 'is_paiddata' being 0 and 'Platform' being 'LinkedIn'
    # condition_exclude_if_true = (
    #     (final_df['is_paiddata'] == 0) &
    #     (final_df['Platform'] == 'LinkedIn') &
    #     (final_df['IS_DARK_POST'] == 'TRUE')
    # )
     
    # # Apply the condition to exclude the specific rows
    # final_df = final_df[~condition_exclude_if_true]
    
    # Continue with any remaining operations on final_df after applying the new filter
        
    condition_exclude_if_true = (
        (final_df['is_paiddata'] == 0) &
        (final_df['Platform'] == 'LinkedIn') &
        (final_df['IS_DARK_POST'] == True)
    )
      
    # Apply the condition to exclude the specific rows
    final_df = final_df[~condition_exclude_if_true]


     
    final_df['Pull Date'] = pd.to_datetime(final_df['Pull Date'])
        
        # Calculate min and max date in 'Pull Date' column
    min_date = final_df['Pull Date'].min()
    max_date = final_df['Pull Date'].max()
        
        # Format the date interval as a string
    date_interval = f"{min_date.strftime('%Y-%m-%d')}_{max_date.strftime('%Y-%m-%d')}"

    destination_key = 'amazon_sprinklr_pull/for_tagging/final_need_to_tagg_'+date_interval
    #destination_key ='amazon_sprinklr_pull/finalmaster/final_need_to_tagg_'+date_interval
    upload_csv_to_s3(final_df, destination_bucket, destination_key)
    
    


    
    return {
        'statusCode': 200,
        'body': json.dumps("success")
    }

     
