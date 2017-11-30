import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.tasks.JavaExec

class OpenCvPlugin implements Plugin<Project> {

    @Override
    void apply(Project target) {
        target.tasks.withType(JavaExec) {
            systemProperty "java.library.path", "/home/obyoxar/Dokumente/RobolabStatistics/RobolabStatistics/lib/opencv"
        }

        target.dependencies {
            compile name: 'opencv-331'
        }

        target.repositories {
            flatDir {
                dirs '/home/obyoxar/Dokumente/RobolabStatistics/RobolabStatistics/lib/opencv/'
            }
        }
    }
}