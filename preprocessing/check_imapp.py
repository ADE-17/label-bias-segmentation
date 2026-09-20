import pandas as pd
ita_df = pd.read_csv('../configs/ita_stats_kmeans.csv')
ita_df['isic_id'] = ita_df['image_path'].apply(lambda x: x.split('/')[-1].split('.')[0])
ima_img_df = pd.read_csv('/path/to/IMA_dataset/img_metadata.csv')
ima_seg_df = pd.read_csv('/path/to/IMA_dataset/seg_metadata.csv')

intersection = set(ita_df['isic_id']).intersection(set(ima_img_df['isic_id']))
print(f'Total ISIC with ITA stats: {len(ita_df)}')
print(f'Total IMA++ images: {len(ima_img_df)}')
print(f'Intersection size: {len(intersection)}')

intersect_seg = ima_seg_df[ima_seg_df['ISIC_id'].isin(intersection)]
print(f'Segmentation masks in intersection: {len(intersect_seg)}')

multiple_annotators = intersect_seg.groupby('ISIC_id').size()
multiple_annotators = multiple_annotators[multiple_annotators > 1]
print(f'Images with multiple annotators in intersection: {len(multiple_annotators)}')
