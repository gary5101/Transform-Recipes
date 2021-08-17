# Project
In this project, we implement a pipeline based on [this paper](https://groups.csail.mit.edu/graphics/xform_recipes/data/xform_paper_sigasia2015.pdf) that reduces the time and energy​ cost of uploading an image and downloading the output images after applying certain​ transformations.​

## Installation

- To install all the requirements, move to the home directory and run
```
pip install -r requirements.txt
```

- To run the code on a sample input and expected output image, go to `src` folder and run
```
bash run.sh <---path-to-input-image---> <---path-to-expected-output--->
```
## Results

Our code works well for transformations, where there is high correlation between input and output images. Few results are shown below:-

| Input Image | Expected Output | Our Output |
|----------|--------------------|--------|
| ![alt text](inputs/inp.jpg "Example Input") | ![alt text](inputs/out.jpg "Given Output")| ![alt text](results/given_example_results/out.png "Our Output")|
| ![alt text](inputs/water_haze.jpg "Haze input") | ![alt text](inputs/water_dehaze.jpg "Dehaze output") | ![alt text](results/dehazing_results/dehaze_out.png "Our Dehaze output") |
| ![alt text](inputs/morning.jpg "morning input") | ![alt text](inputs/night.jpg "night output") | ![alt text](results/time_of_day_results/out.png "Our night output") |
