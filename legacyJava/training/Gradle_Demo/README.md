# Opencv-Project
How to create an opencv-project in this environment

1. Create new Folder "YOURFOLDERNAME" in Training
2. Initialize that folder with Java-Plugin (Either use `gradle init --type java-application` 
which generates some useless stuff that needs revision OR do it manually. See below)
3. Register your Project in settings.gradle.
4. You can now access you new project via Gradle `gradle :training_YOURFOLDERNAME:TASKNAME` or using 
the gradle panel of your IDE.

## Initialize Java Folder
### The manual way
Create a Folder with the following files
* build.gradle
* src/main/java/YOURMAINNAME.java

Write the following into build.gradle

```groovy
apply plugin: 'java'
apply plugin: 'application'
apply plugin: 'opencv' //Leave this out if you don't need opencv

mainClassName = 'YOURMAINNAME'
```
You can start developing your application now and have full opencv support.
### The gradle way
This section is still TODO

## Register Project in Settings gradle
Append the following in the training section of the root `settings.gradle` file.
```groovy
includeTraining 'YOURFOLDERNAME'
```