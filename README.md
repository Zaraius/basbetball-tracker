# basbetball-tracker

We wanted to explore central machine vision topics of identifying an object and its position in 3D space, as well as determining it’s trajectory and estimating some kinematics of objects. We were hoping to frame this as analyzing a basketball as it is shot around the court, identifying its position, trajectory, velocity, and when it goes into a hoop.

Data collection:
...

tried color mask but bas so yolo, trained twice

The first part of this requires us to identify the position of the ball in x and y(maybe z). We first need to find the ball in the camera frame. Our initial approach was to try a color mask to separate the ball and the background but we quickly realized this was an impossible task. The ball wouldn't always have consistent HSV values because of varying lighting and shadows. On top of this the floor and backdrop had similar hues as the ball.

This led us to try using an object identification model. We started with the yolo 11n object identification model and trained on a dataset of basketballs we found on the internet. 
https://universe.roboflow.com/eagle-eye/basketball-1zhpe/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true

After training this model we tested it out on a test video and noticed that it was good at generally finding the basketball, it missed a lot, especially when the ball was far away or slightly covered. We ran a script which generated 50 annotated images and manually fixed their annotations. We split these 50 images back into the train/test/validate(explain this? how yolo models work in general?) and retrained the yolo model. With this new model we are able to consistently get a bounding box for the ball. With the bounding box of the ball we can... segment? find its size??

Basics physics modeling