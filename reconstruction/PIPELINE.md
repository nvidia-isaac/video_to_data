
1. Annotate prompts for human / object using sam2
2. Run sam2 to compute masks
3. Run moge to compute depth
4. Run sam3d using the image/mask from frame of annotated sam2 prompt to get mesh
5. run simplify mesh 
6. run align mesh using depth frame from moge corresponding to frame of annotation (used to create sam3d)
7. run transform mesh using aligned transform and simplified mesh to get mesh as input to foundationpose
8. run foundation pose using mask from annotated frame of object for annotation
9. run foundation pose render overlay
10. run nlf to smpl using person masks and input video
11. run nlf render overlay to render output

