### How to obtain the results in the `./outputs` folder

---

1. Run `./3_1_individual_cases.py` to compile the results of the OSSE individual cases (fully observed, measurement error, sparse cloud, and dense cloud). This script also computes the PSD analysis of the fully observed OSSE.
2. Run `./3_2_evaluation_metrics.py` to visualize statistics for the different OSSE types (time, measurement error, sparse cloud, and dense cloud). For example, it shows the root mean square error of the prediction as a function of the measurement error.
3. Run `./3_3_time_series_computation.py` to generate a dataset for each time step containing the OSSE data, predictions, and samples of the predictions. These datasets are saved as NetCDF files in `./intermediate`. They are used in the remaining scripts of this section.
4. Run `./3_4_time_series_point_analysis.py` to visualize the predictions as a function of time for a single point in the domain.
5. Run `./3_5_time_series_spatial_analysis.py` to compute statistical analyses for a predicted time series and visualize them in two dimensions.
6. Run `./3_6_time_series_to_video.py` to visualize the prediction at each time step. Each step is saved as a frame in `./intermediate`, and the frames are then combined into an `.mp4` video saved in `./outputs`.