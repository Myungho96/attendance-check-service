package hello.hellospring;

import hello.hellospring.repository.*;
import hello.hellospring.service.AttendanceMemberService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.persistence.EntityManager;

@Configuration
public class SpringConfig {


    private EntityManager em;

    @Autowired
    public SpringConfig(EntityManager em){
        this.em = em;
    }

    @Bean
    public AttendanceMemberService attendanceMemberService(){
        return new AttendanceMemberService(attendanceMemberRepository());
    }

    @Bean
    public AttendanceMemberRepository attendanceMemberRepository(){

        return new JpaAttendanceMemberRepository(em);
    }



}