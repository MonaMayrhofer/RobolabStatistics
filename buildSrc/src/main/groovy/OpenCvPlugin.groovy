import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.tasks.JavaExec

class OpenCvPlugin implements Plugin<Project> {

    @Override
    void apply(Project target) {
        target.tasks.withType(JavaExec) {
            systemProperty "java.library.path", target.rootProject.file('lib/opencv').absolutePath
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