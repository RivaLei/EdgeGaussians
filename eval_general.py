import argparse
import os
import pickle
import open3d as o3d
import numpy as np

from plyfile import PlyData
from tqdm.auto import tqdm

import edgegaussians.utils.eval_utils as eval_utils

def main(metrics_pr):
    
    parser = argparse.ArgumentParser(description="evaluate the results")
    parser.add_argument("--dataset", type=str, help="Dataset name", default = "ABC")#必填
    parser.add_argument("--scan_names", type=str, help="Name of the scene")#必填
    parser.add_argument("--use_parametric_edges", action="store_true", help="Use parametric edges fro")#二者二选一,默认用这个
    parser.add_argument("--use_filtered_points", action="store_true", help="Use filtered points from our method")
    parser.add_argument("--visualize", action="store_true", help="Visualize the results")
    parser.add_argument("--version", type=str, default="v4", help="Version of our model")#必填 与config统一
    parser.add_argument("--edge_detector", type=str, default="DexiNed", help="Edge detector used")#必填 与config统一
    parser.add_argument("--scale_points", type=float, default=1.0, help="Scale the our points")
    parser.add_argument("--gt_base_dir", type=str, help="Path to the data directory")#必填
    parser.add_argument("--sample_resolution", type=float, default=0.005, help="Resolution of the sampled points")
    parser.add_argument("--write_metrics", action="store_true", help="Write metrics to file")
    parser.add_argument("--write_metrics_dir", type=str, help="Path to the directory to save metrics", default="metrics/ABC")#与output统一
    
    args = parser.parse_args()
    dataset = args.dataset
    scan_names = args.scan_names
    use_parametric_edges = True if args.use_parametric_edges else False
    visualize = True if args.visualize else False
    use_filtered_points = True if args.use_filtered_points else False
    version = args.version
    edge_detector = args.edge_detector
    write_metrics = True if args.write_metrics else False
    scale_points = args.scale_points
    gt_base_dir = args.gt_base_dir
    
    geometries = []

    output_base_dir = f"output/"+dataset+"/"+version+"_"+edge_detector


    if scan_names == "all":
        scan_names = list(os.listdir(output_base_dir))
    
    else:
        scan_names = scan_names.split(",")
    
    metrics = {}
    str_key = version+"_"+edge_detector
    for scan_name in tqdm(scan_names):
        
        print(f"Evaluating {scan_name}")
        
        eval_dir = os.path.join(output_base_dir, scan_name,"eval_results")
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        metrics[scan_name] = {}

        # read gt if already computed
        ply_path = os.path.join(gt_base_dir, "sampled_pts", f"{scan_name}_{args.sample_resolution}.ply")#abc
        ply_path_2 = os.path.join(gt_base_dir, "edge_points",scan_name, "edge_points.ply")#dtu
        
        if os.path.exists(ply_path):
            pcd_gt = o3d.io.read_point_cloud(ply_path)
            gt_points = np.asarray(pcd_gt.points)
        
        elif os.path.exists(ply_path_2):
            pcd_gt = o3d.io.read_point_cloud(ply_path_2)
            gt_points = np.asarray(pcd_gt.points)
        
        # else compute gt sampled ppoints
        else:
            #其他数据 采样--还没实验过
            _, gt_points, gt_dirs = eval_utils.get_gt_points(scan_name,
                        edge_type="all",
                        interval=0.005,
                        return_direction=False,
                        data_base_dir=gt_base_dir)

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_points)
        pcd_gt.paint_uniform_color([0, 1, 0])
        geometries.append(pcd_gt)

        # Get the estimated points
        pts = None
            
        if use_filtered_points:
            try:
                our_pts_ply_file = os.path.join(output_base_dir, scan_name, "gaussians_filtered.ply")
                if not os.path.exists(our_pts_ply_file):
                    raise Exception("File not found")
                
                plydata = PlyData.read(our_pts_ply_file)
                data = plydata['vertex']
                x = data['x']; y = data['y']; z = data['z']
                pts = np.hstack((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]))

                if pts.shape[0] == 0:
                    raise Exception("No points found")
                pts *= scale_points
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.paint_uniform_color([0, 0, 0])
                geometries.append(pcd)
            except:
                print("Our points not found")

            
        elif use_parametric_edges:
            try:
                
                our_pts_ply_file = os.path.join(output_base_dir, scan_name, f"edge_sampled_points_{args.sample_resolution}.ply")
                
                if os.path.exists(our_pts_ply_file):
                    pcd = o3d.io.read_point_cloud(our_pts_ply_file)
                    pts = np.asarray(pcd.points)
                    pts *= scale_points
                    pcd.paint_uniform_color([0, 0, 0])
                    geometries.append(pcd)
                
                else:
                    our_parametric_edges_file = os.path.join(output_base_dir, scan_name, "parametric_edges.json")
                    print(our_parametric_edges_file)
                    all_curve_points, all_line_points, _, _ = eval_utils.get_pred_points_and_directions(our_parametric_edges_file)   
                    # ipdb.set_trace()
                    pts = np.concatenate([all_curve_points, all_line_points], axis=0)
                    if pts.shape[0] == 0:
                        raise Exception("No points found")
                    
                    pts *= scale_points
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd.paint_uniform_color([0, 0, 1])
                    geometries.append(pcd)
            except:
                print("Our points not found")

        
        if pts is not None:
            chamfer_dist, acc, comp = eval_utils.compute_chamfer_distance(pts.astype(np.float32), gt_points.astype(np.float32))
            # metrics[scan_name]["edgegaussians"] = {"chamfer_dist": chamfer_dist, "acc": acc, "comp": comp}
            metrics[scan_name][str_key] = {"chamfer_dist": chamfer_dist, "acc": acc, "comp": comp}
            # print("edgegaussians")
            # print(f"Chamfer distance: {chamfer_dist}")
            # print(f"Acc: {acc}")
            # print(f"Comp: {comp}")

            metrics_pr = eval_utils.compute_precision_recall_IOU(pts.astype(np.float32), gt_points.astype(np.float32), metrics_pr, thresh_list=[0.005, 0.01, 0.02], edge_type="all")
            # print("Recall @ 5mm",metrics_pr["recall_0.005"][-1], "Precision @ 5mm", metrics_pr["precision_0.005"][-1])
            # print("Recall @ 10mm",metrics_pr["recall_0.01"][-1], "Precision @ 10mm", metrics_pr["precision_0.01"][-1])
            # print("Recall @ 20mm",metrics_pr["recall_0.02"][-1], "Precision @ 20mm", metrics_pr["precision_0.02"][-1])

        if visualize:
            
            o3d.visualization.draw_geometries(geometries)

            #保存图片为全黑
            
            # # 创建一个可视化窗口
            # vis = o3d.visualization.Visualizer()
            # vis.create_window()

            # # 添加几何体
            # for geometry in geometries:
            #     vis.add_geometry(geometry)
            
            # # 设置相机位置--不设置 则为默认视角
            # # view_control = vis.get_view_control()
            # # view_control.set_front([0, 0, -1])
            # # view_control.set_lookat([0, 0, 0])
            # # view_control.set_up([0, -1, 0])
            # # view_control.set_zoom(0.8)

            # # 渲染几何体
            # vis.poll_events()
            # vis.update_renderer()

            # # 保存影像
            # image_path = os.path.join(eval_dir, "gt_ours.png")
            # vis.capture_screen_image(image_path)

            # # 关闭窗口
            # vis.destroy_window()

            # print(f"Image saved to {image_path}")
            
        geometries = []
    
    for key in metrics_pr.keys():
        if len(metrics_pr[key]) == 0:
            continue
        else:
            mean_val = np.mean(np.array(metrics_pr[key]))
            print(f"{key}: {mean_val}")

    vals = {}
    for key in metrics.keys():

        if str_key in metrics[key].keys():
            ours = metrics[key][str_key]
            for key2 in ours.keys():
                if key2 not in vals.keys():
                    vals[key2] = [ours[key2]]
                else:
                    vals[key2] += [ours[key2]]

    for key in vals.keys():
        mean_val = np.mean(np.array(vals[key]))
        print(f"{key}: {mean_val}")
    
    if write_metrics:
        write_metrics_dir = eval_dir
        print(f"Writing metrics to {write_metrics_dir}...")
        if not os.path.exists(write_metrics_dir):
            os.makedirs(write_metrics_dir)
        
        app_str = ""    
        if use_filtered_points:
            app_str="filtered_points"
        elif use_parametric_edges:
            app_str="parametric_edges"

        
        #将metrics_pr写入txt文本中
        with open(os.path.join(write_metrics_dir, app_str+"_pr.txt"), "w") as f:
            for key in metrics_pr.keys():
                if len(metrics_pr[key]) == 0:
                    continue
                else:
                    mean_val = np.mean(np.array(metrics_pr[key]))
                    f.write(f"{key}: {mean_val}\n")
                    
        #将metrics写入txt文本中
        with open(os.path.join(write_metrics_dir, app_str+"_acc_comp_chamfer.txt"), "w") as f:
            for key in vals.keys():
                mean_val = np.mean(np.array(vals[key]))
                f.write(f"{key}: {mean_val}\n")
                
        
        
        
        with open(os.path.join(write_metrics_dir, app_str+"_pr.pkl"), "wb") as f:
            pickle.dump(metrics_pr, f)
        
        with open(os.path.join(write_metrics_dir, app_str+"_acc_comp_chamfer.pkl"), "wb") as f:
            pickle.dump(metrics, f)

if __name__ == "__main__":
    metrics_pr = {
        "chamfer": [],
        "acc": [],
        "comp": [],
        "comp_curve": [],
        "comp_line": [],
        "acc_curve": [],
        "acc_line": [],
        "precision_0.01": [],
        "recall_0.01": [],
        "fscore_0.01": [],
        "IOU_0.01": [],
        "precision_0.02": [],
        "recall_0.02": [],
        "fscore_0.02": [],
        "IOU_0.02": [],
        "precision_0.005": [],
        "recall_0.005": [],
        "fscore_0.005": [],
        "IOU_0.005": [],
    }

    main(metrics_pr)