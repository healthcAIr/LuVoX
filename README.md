# Workflow-centred open-source fully automated lung volumetry in chest CT (LuVoX)

This is the code for the paper

[**Workflow-centred open-source fully automated lung volumetry in chest CT**](https://doi.org/10.1016/j.crad.2019.08.010)  
F. Jungmann*, S. Brodehl*, R. Buhl, P. Mildenberger, E. Schömer, C. Düber, D. Pinto dos Santos
(* equal contribution)  
Clinical Radiology, Volume 75, Issue 1, 2020, Pages 78.e1-78.e7

Our open source algorithm allows fast and fully automated calculation of lung volume in multidetector computed tomography.
The paper shows that lung volume measured by CT correlated significantly with pulmonary function testing.
The integration of the algorithm into the clinical workflow offers measurements at the start of the reporting process without manual interaction.

![The image contains overlays in lung window settings to check the plausibility of automatically measuring the lung. The axial images (lung window with overlay) and 3D reconstruction show perfect identification of lung parenchyma in a patient after lobectomy.](example_overlay_and_3D.png)

We provide:

- Code to [run the algorithm on CT images](#running-on-new-images)
- Code to [produce reports from CT images](#creating-reports)

If you find this work useful in your research, please cite:
```
@article{JUNGMANN202078.e1,
title = "Workflow-centred open-source fully automated lung volumetry in chest CT",
author = "F. Jungmann and S. Brodehl and R. Buhl and P. Mildenberger and E. Schömer and C. Düber and D. Pinto dos Santos"
journal = "Clinical Radiology",
volume = "75",
number = "1",
pages = "78.e1 - 78.e7",
year = "2020",
issn = "0009-9260",
doi = "https://doi.org/10.1016/j.crad.2019.08.010",
url = "http://www.sciencedirect.com/science/article/pii/S0009926019305197",
}
```

## Getting Started

### Prerequisites

The current version requires in particular the following libraries / versions.
See [requirements.txt](requirements.txt) for the full list of requirements.

* [Python 3](https://www.python.org/downloads/), version `2.x` might work, no guarantees, no support.
* [pydicom 1.0.0](https://github.com/pydicom/pydicom) or newer.
* [OpenCV 3](https://opencv.org) python bindings or newer, version `2.x` might work.

The easiest way to install those dependencies is by using the [requirements.txt](requirements.txt) file with `pip3`.
```commandline
pip3 install -r requirements.txt
```

## Acknowledgments

* *Guido Zuidhof's* [Full Preprocessing Tutorial](https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial)
for [Data Science Bowl 2017](https://www.kaggle.com/c/data-science-bowl-2017) for inspiration on data preprocessing (with Python).
* [*@rhaxton*](https://github.com/rhaxton) for inspiration on [in-memory data decompression](https://github.com/pydicom/pydicom/issues/219) paired with [pydicom](https://github.com/pydicom/pydicom).
