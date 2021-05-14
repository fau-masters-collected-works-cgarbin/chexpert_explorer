import chexpert_dataset as cxd
import chexpert_statistics as cxs

cxdata = cxd.CheXpertDataset()

# Count of patients and images in the training and validation datasets
# Long format, easier to index in code
stats = cxs.patient_study_image_count(cxdata.df)
print(stats)

# Wide format, better for documentation
stats = stats.unstack().droplevel(0, axis='columns')
print(stats)
