import pandas as pd
import numpy as np
import os

def prep():
    print("Loading metadata...")
    ita_df = pd.read_csv('configs/ita_stats_kmeans.csv')
    ita_df['isic_id'] = ita_df['image_path'].apply(lambda x: x.split('/')[-1].split('.')[0])
    
    seg_df = pd.read_csv('/path/to/IMA_dataset/seg_metadata.csv')
    
    # 1. Filter to intersection
    intersection = set(ita_df['isic_id']).intersection(set(seg_df['ISIC_id']))
    ita_df = ita_df[ita_df['isic_id'].isin(intersection)]
    seg_df = seg_df[seg_df['ISIC_id'].isin(intersection)]
    
    valid_ids = list(intersection)
    print(f"Found {len(valid_ids)} images in intersection.")
    
    # 3. Create mapping
    records = []
    
    img_dir = '/path/to/IMA_dataset'
    
    # Map skin tones to gender indices
    skin_tone_map = {'Very Light': 0, 'Light': 1, 'Intermediate': 2, 'Tan': 3}
    
    for isic_id in valid_ids:
        # Get skin tone
        skin_tone_str = ita_df[ita_df['isic_id'] == isic_id].iloc[0]['category']
        if skin_tone_str not in skin_tone_map:
            continue
        gender = skin_tone_map[skin_tone_str]
        
        image_path = os.path.join(img_dir, f"{isic_id}.jpg")
        
        # Get all segs for this image
        img_segs = seg_df[seg_df['ISIC_id'] == isic_id]
        
        # Determine Clean mask -> A01 if available, else STAPLE, else MV, else any
        clean_mask_row = None
        if 'A01' in img_segs['annotator'].values:
            clean_mask_row = img_segs[img_segs['annotator'] == 'A01'].iloc[0]
        elif 'ST' in img_segs['annotator'].values:
            clean_mask_row = img_segs[img_segs['annotator'] == 'ST'].iloc[0]
        elif 'MV' in img_segs['annotator'].values:
            clean_mask_row = img_segs[img_segs['annotator'] == 'MV'].iloc[0]
        else:
            clean_mask_row = img_segs.iloc[0]
            
        clean_mask_path = os.path.join(img_dir, clean_mask_row['seg_filename'])
        
        # Determine Biased mask -> Any other human annotator, or fallback to STAPLE/MV if no other human
        other_humans = img_segs[(img_segs['annotator'].str.startswith('A')) & (img_segs['annotator'] != 'A01')]
        
        if len(other_humans) > 0:
            # Try to pick S2 (Novice)
            s2_segs = other_humans[other_humans['skill_level'] == 'S2']
            if len(s2_segs) > 0:
                biased_mask_row = s2_segs.iloc[0]
            else:
                biased_mask_row = other_humans.iloc[0]
        else:
            # Fallback to STAPLE or MV if no other human
            if 'ST' in img_segs['annotator'].values:
                biased_mask_row = img_segs[img_segs['annotator'] == 'ST'].iloc[0]
            elif 'MV' in img_segs['annotator'].values:
                biased_mask_row = img_segs[img_segs['annotator'] == 'MV'].iloc[0]
            else:
                # Same as clean (should rarely happen)
                biased_mask_row = clean_mask_row
                
        biased_mask_path = os.path.join(img_dir, biased_mask_row['seg_filename'])
        
        records.append({
            'isic_id': isic_id,
            'image_path': image_path,
            'clean_mask_path': clean_mask_path,
            'biased_mask_path': biased_mask_path,
            'gender': gender,
            'category': skin_tone_str
        })
        
    out_df = pd.DataFrame(records)
    out_df.to_csv('configs/imapp_processed.csv', index=False)
    print(f"Saved processed dataset mapping to configs/imapp_processed.csv with {len(out_df)} records.")

if __name__ == '__main__':
    prep()
