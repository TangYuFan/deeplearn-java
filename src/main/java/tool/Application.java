package tool;


import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;


@SpringBootApplication
@ComponentScan(value="tool.*")
@RestController
@RequestMapping("/app")
public class Application {

	@RequestMapping("/1")
	public String test1(){
		return "1";
	}

	public static void main(String[] args) {
		SpringApplication.run(Application.class, args);
	}

}
