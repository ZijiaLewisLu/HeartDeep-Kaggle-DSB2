# HeartDeepLearning

Based on Kaggle's Second Data Science Bowl, we are trying to analyze the healthy condition of patients, using Deep Network.

MRI is used to slice out the structure of hearts from various angles, and outputs are formated into DICOM images. Short Axis is the special angle from which the cross section of Left Ventricle(LV) can be seen. Given a series of Short Axis from different heights at the same time, doctors are able to calculate the volumn of LV. Then doctors will compare the volumns at systole and disatole to inspect the vitality of hearts.

As the process is time-costing, the task for us is to automatically conduct this it to reduce doctors' work load. Different from the FCN suggested by Kaggle's tutorial, Time Series will be added into our solution. In other words, the net will not look at all images from different time, calculate all the volumn, and choose the biggest as the disatole and smallest as systole. Instead, it should be able to know when are the disatole and systole after having a rough glimpse at all images and only calculate on right images. Thus, the duration can be reduced.

This project is still in process. It will be high appreciated if you want to offer some suggestions!

