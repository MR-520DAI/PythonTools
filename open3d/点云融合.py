import numpy as np
import open3d as o3d

intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

pose = []
pose0 = np.array([[1.,0.,0.,0.],
                  [0.,1.,0.,0.],
                  [0.,0.,1.,0.],
                  [0.,0.,0.,1.]])
pose1 = np.array([[0.99326466389121126,-0.019485360480441097,-0.11421746010146851,0.14882547641641353],
                [0.021240089437541521,0.99967403631303031,0.014166147051987699,-0.19981651554415372],
                [0.11390419687513019,-0.01649672235811022,0.99335476647855003,0.32346341724021349],
                [0.0,0.0,0.0,1.0]])
pose2 = np.array([[0.92371809918580206,-0.141029707656016,0.35616779022678413,0.73205086771528249],
                [0.15393438961594497,0.98804872085201501,-0.0079955560324353782,-0.6247632628364187],
                [-0.35078351861245294,0.062212111209633325,0.93438780829411339,1.2529881511129219],
                [0.,0.,0.,1.]])
pose3 = np.array([[0.66350400202980353,-0.17020547686910448,0.7285551008222978,0.94955449809796855],
                [0.3072552633784757,0.94986326973379542,-0.057913486657677947,-0.63501034234745712],
                [-0.68217053763456437,0.26227821955775227,0.6825346094733582,1.0241711584623197],
                [0.,0.,0.,1.]])
pose.append(pose0)
pose.append(pose1)
pose.append(pose2)
pose.append(pose3)

pcd = []
pcd0 = o3d.io.read_point_cloud("data\\fragment_000.ply")
pcd1 = o3d.io.read_point_cloud("data\\fragment_001.ply")
pcd2 = o3d.io.read_point_cloud("data\\fragment_002.ply")
pcd3 = o3d.io.read_point_cloud("data\\fragment_003.ply")
pcd.append(pcd0)
pcd.append(pcd1)
pcd.append(pcd2)
pcd.append(pcd3)

pcd_all = o3d.geometry.PointCloud()
for i in range(len(pose)):
    pcd_all+=pcd[i].transform(pose[i])


o3d.visualization.draw_geometries([pcd_all])