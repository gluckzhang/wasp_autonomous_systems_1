# Sensor fusion with GPS and IMU

The requirements of this assignment can be read [on this page](https://kth.instructure.com/courses/4962/assignments/15984).  

Our goal is to evaluate the effects of GPS signal outage on the navigation solution and use a Kalman filter to optimize the sensor fusion.  

Short introduction to the 4 tasks:  

- Task 1: Use the functions errorgrowth.m and Nav eq.m to evaluate how the position error grows with time
- Task 2: Modify the code to simulate a GNSS-receiver outage from 200 seconds and onward
- Task 3: Implement support for non-holonomic motion constraints
- Task 4: Implement support for speedometer measurements