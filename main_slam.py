#!/usr/bin/env -S python3 -O
"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""
import matplotlib.pyplot as plt
import cv2
import time 
import os
import sys
import platform 
import evaluate

from config import Config

from slam import Slam, SlamState
from slam_plot_drawer import SlamPlotDrawer
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset import dataset_factory, SensorType
from trajectory_writer import TrajectoryWriter

from evaluate import evaluate_ate, evaluate_rpe


if platform.system()  == 'Linux':     
    from display2D import Display2D  #  !NOTE: pygame generate troubles under macOS!

from viewer3D import Viewer3D
from utils_sys import getchar, Printer 
from utils_img import ImgWriter

from feature_tracker_configs import FeatureTrackerConfigs

from loop_detector_configs import LoopDetectorConfigs

from depth_estimator_factory import depth_estimator_factory, DepthEstimatorType
from utils_depth import img_from_depth, filter_shadow_points

from config_parameters import Parameters  

from rerun_interface import Rerun

import traceback

from utilities.utils_print import GlobalPrinter
time.sleep(1)
GlobalPrinter.get_instance().set_log_file("logs/main.log")
time.sleep(1)
GlobalPrinter.get_instance().set_print_to_terminal(False)
time.sleep(1)
compare_gt = False

def initiate_quit(slam, plot_drawer, display2d, viewer3D, trajectory_writer, output_dir,gt_file = None,traj_file = None):
    # plot_drawer.save_plots(output_dir)
    slam.quit()
    plot_drawer.quit(output_dir)
    for component in [display2d, viewer3D]:
        if component is not None:
            component.quit()
    trajectory_writer.close_file()
    
    if compare_gt:
        if gt_file is not None and traj_file is not None:
            ate_results.append(evaluate_ate(gt_file, traj_file))
            rpe_results.append(evaluate_rpe(gt_file, traj_file))


ate_results = []
rpe_results = []



if __name__ == "__main__":
                               
    config = Config()

    # OPTIONAL: Define how many times we want to run each dataset-feature combo
    num_iterations = 5  # >= 1

    #Define the list of feature-tracker configs you want to test.
    # Select your tracker configuration (see the file feature_tracker_configs.py) 
    # FeatureTrackerConfigs: SHI_TOMASI_ORB, FAST_ORB, ORB, ORB2, ORB2_FREAK, ORB2_BEBLID, BRISK, AKAZE, 
        #FAST_FREAK, SIFT, ROOT_SIFT, SURF, KEYNET, SUPERPOINT, FAST_TFEAT, CONTEXTDESC, LIGHTGLUE,
        #XFEAT, XFEAT_XFEAT
        #ORB2,ORB,XFEAT
    # WARNING: At present, SLAM does not support LOFTR and other "pure" image matchers (further details in the commenting notes about LOFTR in feature_tracker_configs.py).
    feature_configs_to_test = [
        # FeatureTrackerConfigs.ORB,
        # FeatureTrackerConfigs.ORB2,
        # FeatureTrackerConfigs.SHI_TOMASI_ORB,
        FeatureTrackerConfigs.FAST_ORB,
        FeatureTrackerConfigs.SHI_TOMASI_FREAK, # this is not tested
        FeatureTrackerConfigs.FAST_FREAK, # this is not tested
        FeatureTrackerConfigs.BRISK,
        FeatureTrackerConfigs.KAZE,
        FeatureTrackerConfigs.AKAZE,
        FeatureTrackerConfigs.ROOT_SIFT,
        FeatureTrackerConfigs.SUPERPOINT,
        FeatureTrackerConfigs.XFEAT,
        FeatureTrackerConfigs.XFEAT_XFEAT,
        FeatureTrackerConfigs.DELF,
        FeatureTrackerConfigs.D2NET,
        FeatureTrackerConfigs.R2D2,
        FeatureTrackerConfigs.LFNET,
        FeatureTrackerConfigs.CONTEXTDESC,
        FeatureTrackerConfigs.KEYNET,
        FeatureTrackerConfigs.DISK,
        FeatureTrackerConfigs.KEYNETAFFNETHARDNET,
        FeatureTrackerConfigs.SIFT,
        FeatureTrackerConfigs.ALIKED,
        FeatureTrackerConfigs.DISK,
        FeatureTrackerConfigs.LIGHTGLUE_ALIKED,
        FeatureTrackerConfigs.LIGHTGLUESIFT,
        FeatureTrackerConfigs.LIGHTGLUE_DISK,
        # FeatureTrackerConfigs.LIGHTGLUESIFT,
        FeatureTrackerConfigs.ORB2_BEBLID,
    ]
    
    base_log_path = 'logs'
    
    startfrom  = {
        'ft_type': FeatureTrackerConfigs.SHI_TOMASI_FREAK,
        'dataset_name': 'MH02',
        'iteration_idx': 0
    }
    
    print('Slam System Started')

    # set up flags to skip over feature tracker configs, datasets, and iterations, 
    # if false the loop will skip over that feature tracker config, dataset, or iteration until it reaches the startfrom values
    Feat_tracker = False
    _dataset = False
    _iteration = False
    
    # 2) Now loop over each feature config
    for feature_tracker_config in feature_configs_to_test:
        
        if (feature_tracker_config != startfrom['ft_type']) and (not Feat_tracker):
            print(f'Skipping feature tracker config: {feature_tracker_config["detector_type"]}_{feature_tracker_config["descriptor_type"]}')
            continue
        else:
            Feat_tracker = True

        # 3) Loop over each dataset name if multiple_datasets == True, or just the single dataset name
        for dataset_name in config.dataset_settings['dataset_names']:
            
            if (dataset_name != startfrom['dataset_name']) and (not _dataset):
                print(f'Skipping dataset: {dataset_name}')
                continue
            else:
                _dataset = True

            Printer.green(f'Running SLAM on dataset: {dataset_name} with feature config: {feature_tracker_config["detector_type"]}')

            # Because we want to re-instantiate the dataset for each sequence, we set config.dataset_settings['name']
            config.dataset_settings['name'] = dataset_name
            
            if config.dataset_type == 'EUROC_DATASET':
                gt_file = os.path.join(config.dataset_settings['base_path'], dataset_name,'/mav0/state_groundtruth_estimate0/data.tum')
            

            ate_results = []
            rpe_results = []
            txt = ''
            
            #run multiple iterations for the same (dataset, feature) pair
            for iteration_idx in range(num_iterations):
                
                if (iteration_idx < startfrom['iteration_idx']) and (not _iteration):
                    print(f'Skipping iteration {iteration_idx}')
                    continue
                else:
                    _iteration = True
                
                # pass
                
                # (a) Override the trajectory filename to reflect feature/dataset/iteration
                #     e.g. {base_log_path}/ORB2_BEBLID/MH04/0/trajectory.txt
                # NOTE: We assume config.trajectory_settings is a dict with keys: 'filename', 'save_trajectory', 'format_type'
                output_dir = f"{base_log_path}/{feature_tracker_config['detector_type']}_{feature_tracker_config['descriptor_type']}/{dataset_name}/{iteration_idx}"
                os.makedirs(output_dir, exist_ok=True)
                
                # Set the log file for this iteration
                GlobalPrinter.get_instance().set_print_to_terminal(False)
                GlobalPrinter.get_instance().set_log_file(f"{output_dir}/log.txt")
                
                Printer.yellow(f'--- Iteration {iteration_idx+1}/{num_iterations} ---')
                time.sleep(2)
                
                # We build a full path for the trajectory file:
                traj_name = f"{feature_tracker_config['detector_type']}_{dataset_name}_{iteration_idx}_trajectory.txt"
                traj_file = os.path.join(output_dir, traj_name)
                config.trajectory_settings['filename'] = traj_file
                
                #open trajectory file
                trajectory_writer = None
                if config.trajectory_settings['save_trajectory']:
                    trajectory_writer = TrajectoryWriter(
                        format_type=config.trajectory_settings['format_type'], 
                        filename=config.trajectory_settings['filename']
                        )
                    trajectory_writer.open_file()

                dataset = dataset_factory(config)
                groundtruth = groundtruth_factory(config.dataset_settings)
                camera = PinholeCamera(config)
    
                num_features=2000 
                if config.num_features_to_extract > 0:  # override the number of features to extract if we set something in the settings file
                    num_features = config.num_features_to_extract
                feature_tracker_config['num_features'] = num_features
                Printer.green('feature_tracker_config: ',feature_tracker_config)    
    
                # Select your loop closing configuration (see the file loop_detector_configs.py). Set it to None to disable loop closing. 
                # LoopDetectorConfigs: DBOW2, DBOW3, IBOW, OBINDEX2, VLAD, HDC_DELF, SAD, ALEXNET, NETVLAD, COSPLACE, EIGENPLACES  etc.
                # NOTE: under mac, the boost/text deserialization used by DBOW2 and DBOW3 may be very slow.
                # loop_detection_config = LoopDetectorConfigs.DBOW3
                loop_detection_config = None # Keep this at None because I don't need to evaluate loop detectors
                Printer.green('loop_detection_config: ',loop_detection_config)
        
                # Select your depth estimator in the front-end (EXPERIMENTAL, WIP)
                depth_estimator = None # keep this at None
                if Parameters.kUseDepthEstimatorInFrontEnd:
                    Parameters.kVolumetricIntegrationUseDepthEstimator = False  # Just use this depth estimator in the front-end
                    # Select your depth estimator (see the file depth_estimator_factory.py)
                    # DEPTH_ANYTHING_V2, DEPTH_PRO, DEPTH_RAFT_STEREO, DEPTH_SGBM, etc.
                    depth_estimator_type = DepthEstimatorType.DEPTH_PRO
                    max_depth = 20
                    depth_estimator = depth_estimator_factory(depth_estimator_type=depth_estimator_type, max_depth=max_depth,
                                                            dataset_env_type=dataset.environmentType(), camera=camera) 
                    Printer.green(f'Depth_estimator_type: {depth_estimator_type.name}, max_depth: {max_depth}')       
                
                # create SLAM object
                slam = Slam(camera, feature_tracker_config, loop_detection_config, dataset.sensorType(), environment_type=dataset.environmentType()) # groundtruth not actually used by Slam class
                slam.set_viewer_scale(dataset.scale_viewer_3d)
                time.sleep(1) # to show initial messages 
    
                # load system state if requested         
                if config.system_state_load: 
                    slam.load_system_state(config.system_state_folder_path)
                    viewer_scale = slam.viewer_scale() if slam.viewer_scale()>0 else 0.1  # 0.1 is the default viewer scale
                    print(f'viewer_scale: {viewer_scale}')
                    slam.set_tracking_state(SlamState.INIT_RELOCALIZE)

                viewer3D = Viewer3D(scale=dataset.scale_viewer_3d)
                if groundtruth is not None:
                    gt_traj3d, gt_timestamps = groundtruth.getFull3dTrajectory()
                    if viewer3D is not None:
                        viewer3D.set_gt_trajectory(gt_traj3d, gt_timestamps, align_with_scale=dataset.sensor_type==SensorType.MONOCULAR)
    
                if platform.system() == 'Linux':    
                    display2d = Display2D(camera.width, camera.height)  # pygame interface 
                else: 
                    display2d = None  # enable this if you want to use opencv window

                plot_drawer = SlamPlotDrawer(slam, viewer3D,output_dir)
    
                img_writer = ImgWriter(font_scale=0.7)

                do_step = False      # proceed step by step on GUI 
                do_reset = False     # reset on GUI 
                is_paused = False    # pause/resume on GUI 
                is_map_save = False  # save map on GUI
                
                key_cv = None
                        
                img_id = 0  #180, 340, 400   # you can start from a desired frame id if needed 
                
                
                time_to_stop = 0 
                max_time = 5 #max time to wait for new images else loop will break
                while True:
                    
                    img, img_right, depth = None, None, None    
                    
                    if do_step:
                        Printer.orange('do step: ', do_step)
                        
                    if do_reset: 
                        Printer.yellow('do reset: ', do_reset)
                        slam.reset()
                        
                    if not is_paused or do_step:
                    
                        if dataset.isOk():
                            print('..................................')               
                            img = dataset.getImageColor(img_id)
                            depth = dataset.getDepth(img_id)
                            img_right = dataset.getImageColorRight(img_id) if dataset.sensor_type == SensorType.STEREO else None
                        
                        if img is not None:
                            timestamp = dataset.getTimestamp()          # get current timestamp 
                            next_timestamp = dataset.getNextTimestamp() # get next timestamp 
                            frame_duration = next_timestamp-timestamp if (timestamp is not None and next_timestamp is not None) else -1

                            print(f'image: {img_id}, timestamp: {timestamp}, duration: {frame_duration}') 
                            
                            time_start = None 
                            if img is not None:
                                time_start = time.time()    
                                
                                if depth is None and depth_estimator is not None:
                                    depth_prediction = depth_estimator.infer(img, img_right)
                                    if Parameters.kDepthEstimatorRemoveShadowPointsInFrontEnd:
                                        depth = filter_shadow_points(depth_prediction)
                                    else: 
                                        depth = depth_prediction
                                    depth_img = img_from_depth(depth_prediction, img_min=0, img_max=50)
                                    cv2.imshow("depth prediction", depth_img)
                                    
                                try:        
                                    slam.track(img, img_right, depth, img_id, timestamp)  # main SLAM function 
                                    
                                except Exception as e:
                                    Printer.error('Error in SLAM track()')
                                    print(traceback.format_exc())
                                    cv2.imwrite(os.path.join(output_dir, f"{img_id}_err_image.png"), img)
                                    initiate_quit(slam, plot_drawer, display2d, viewer3D, trajectory_writer, output_dir)
                                    plot_drawer = None
                                    break
                                                
                                # 3D display (map display)
                                if viewer3D is not None:
                                    viewer3D.draw_map(slam)

                                img_draw = slam.map.draw_feature_trails(img)
                                img_writer.write(img_draw, f'id: {img_id}', (30, 30))
                                
                                # 2D display (image display)
                                if display2d is not None:
                                    display2d.draw(img_draw)
                                else: 
                                    cv2.imshow('Camera', img_draw)
                                
                                # draw 2d plots
                                plot_drawer.draw(img_id)
                                    
                            if trajectory_writer is not None and slam.tracking.cur_R is not None and slam.tracking.cur_t is not None:
                                trajectory_writer.write_trajectory(slam.tracking.cur_R, slam.tracking.cur_t, timestamp)
                                
                            if time_start is not None: 
                                duration = time.time()-time_start
                                if(frame_duration > duration):
                                    time.sleep(frame_duration-duration) 
                                
                            img_id += 1 
                        else: 
                            time.sleep(0.1)
                            time_to_stop += 0.1
                            if time_to_stop > max_time:
                                time_to_stop = 0
                                initiate_quit(slam, plot_drawer, display2d, viewer3D, trajectory_writer, output_dir) 
                                plot_drawer = None
                                break
                            
                        # 3D display (map display)
                        if viewer3D is not None:
                            viewer3D.draw_dense_map(slam)  
                                        
                    else:
                        time.sleep(0.1)                                 
                    
                    # get keys 
                    key = plot_drawer.get_key()
                    if display2d is None:
                        key_cv = cv2.waitKey(1) & 0xFF   
                        
                    # if key != '' and key is not None:
                    #     print(f'key pressed: {key}') 
                    
                    # manage interface infos  
                    
                    if slam.tracking.state==SlamState.LOST:
                        if display2d is None:  
                            #key_cv = cv2.waitKey(0) & 0xFF   # useful when drawing stuff for debugging
                            key_cv = cv2.waitKey(500) & 0xFF                                 
                        else: 
                            #getchar()
                            time.sleep(0.5)
                            
                    if is_map_save:
                        slam.save_system_state(config.system_state_folder_path)
                        dataset.save_info(config.system_state_folder_path)
                        Printer.green('uncheck pause checkbox on GUI to continue...\n')        
                    
                    if viewer3D is not None:
                        is_paused = viewer3D.is_paused()    
                        is_map_save = viewer3D.is_map_save() and is_map_save == False 
                        do_step = viewer3D.do_step() and do_step == False  
                        do_reset = viewer3D.reset() and do_reset == False  
                                            
                    if key == 'q' or (key_cv == ord('q')):
                        initiate_quit(slam, plot_drawer, display2d, viewer3D, trajectory_writer, output_dir)
                        plot_drawer = None
                        break
                        
                
                
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                GlobalPrinter.get_instance().set_print_to_terminal(True)

            
                        
            # if compare_gt:
            #     txt = "#_pair_ate,mean_ate,median_ate,std_ate,min_ate,max_ate,"
            #     txt += "#_pairs_rpe_tran,rmse_rpe_tran,mean_rpe_tran,median_rpe_tran,std_rpe_tran,min_rpe_tran,max_rpe_tran,rmse_rpe_rot,mean_rpe_rot,median_rpe_rot,std_rpe_rot,min_rpe_rot,max_rpe_rot,\n"
                
            #     for i in range(num_iterations):
            #         ate_result = ate_results[i]
            #         rpe_results = rpe_results[i]
            #         txt += f"{ate_result['compared_pose_pairs']},{ate_result['absolute_translational_error']['rmse']},{ate_result['absolute_translational_error']['mean']},{ate_result['absolute_translational_error']['median']},{ate_result['absolute_translational_error']['std']},{ate_result['absolute_translational_error']['min']},{ate_result['absolute_translational_error']['max']},"
            #         txt += f"{rpe_results['compared_pose_pairs']},{rpe_results['translational_error']['rmse']},{rpe_results['translational_error']['mean']},{rpe_results['translational_error']['median']},{rpe_results['translational_error']['std']},{rpe_results['translational_error']['min']},{rpe_results['translational_error']['max']},{rpe_results['rotational_error']['rmse']},{rpe_results['rotational_error']['mean']},{rpe_results['rotational_error']['median']},{rpe_results['rotational_error']['std']},{rpe_results['rotational_error']['min']},{rpe_results['rotational_error']['max']},\n"
                
            #     dir = f"{base_log_path}/{feature_tracker_config['detector_type']}/{dataset_name}/"
                
            #     with open(f"{dir}/results.txt", "w") as f:
            #         f.write(txt)
                
                
                
                
                