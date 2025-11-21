# MSAAI521
USD MS AAI - 521 - Fall 2025 - Final Project, Team 2

You need to get the data from https://www.nuscenes.org/nuscenes#download. Get: 

Full dataset (v1.0) -> Mini (3.88GB).

Create a build/data/raw folder at the root of the project and place the unzipped files in there. It should be:

```
MSAAI521/
├── build/
│   ├── data/
│       └── raw/
│           └── v1.0-mini/
│               ├── samples/
│               ├── sweeps/
│               └── v1.0-mini/
│                   └── ...
├── src/
│   └── ...
├── demo_notebook.ipynb
└── README.md

```

Dont forget to run:

```
pip install -r requirements.txt
```

Currently all it does is explore the data and create visualizations. I wasn't able to get the data conversion to BEV (Bird's Eye View) images to align with the annotations so all that code is commented out in main.py.

The code can be both run using the VSCode debugger by executing main.py directly or through the notebook.
