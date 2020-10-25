package com.ca.group1.plp.controller;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.ModelAndView;

import java.util.concurrent.atomic.AtomicLong;

@ComponentScan
@Controller
public class HomeController {
    @Value("${spring.application.name}")
    String appName;

//    @RequestMapping(value="/home")
//    public String home(){
//        System.out.println("redirect to home page!");
//        return "index";
//    }
//
//    @RequestMapping(value="/home/page")
//    @ResponseBody
//    public ModelAndView goHome(){
//        System.out.println("go to the home page!");
//        ModelAndView mode = new ModelAndView();
//        mode.addObject("name", "zhangsan");
//        mode.setViewName("index");
//        return mode;
//    }

    @GetMapping("/home")
    public String homePage(Model model) {
        model.addAttribute("appName", appName);
        return "home";
    }
}
