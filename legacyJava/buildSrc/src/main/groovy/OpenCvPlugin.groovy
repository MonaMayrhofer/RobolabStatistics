import org.apache.tools.ant.taskdefs.condition.Os
import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.tasks.JavaExec

class OpenCvPlugin implements Plugin<Project> {

    @Override
    void apply(Project target) {
        target.tasks.withType(JavaExec) {
            def libFolder
            if(Os.isFamily(Os.FAMILY_WINDOWS)){
                if(Os.isArch("amd64")){
                    libFolder = target.rootProject.file('lib/opencv/x64')
                }else{
                    //Untested
                    libFolder = target.rootProject.file('lib/opencv/x86')
                }
            }else{
                libFolder = target.rootProject.file('lib/opencv')
            }
            systemProperty "java.library.path",libFolder.absolutePath
        }

        target.dependencies {
            compile name: 'opencv-331'
        }

        target.repositories {
            flatDir {
                dirs target.rootProject.file('lib/opencv').absolutePath
            }
        }
    }
}