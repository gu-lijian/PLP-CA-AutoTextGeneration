package com.ca.group1.plp.controller;

import com.ca.group1.plp.pojo.Tweet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.ModelAndView;

import java.util.concurrent.atomic.AtomicLong;

@RestController
public class TweetController {

    private static Logger log = LoggerFactory.getLogger(TweetController.class);
    private static final String template = "inputText is: %s category is: %s emotion is: %s!";
    private final AtomicLong counter = new AtomicLong();
    @GetMapping("/generateTweet")
    public String generateTweet(@RequestParam(value = "inputText", defaultValue = "") String inputText,
                                @RequestParam(value = "topic", defaultValue = "") String topic,
                                @RequestParam(value = "emotion", defaultValue = "") String emotion,
                                ModelAndView mv) {
        // AI magic here
//		return String.format("Hello %s!", "I love cats because I love my home and little by little they become its visible soul");
        System.out.println("inputText is: " + inputText + " " + "topic is: " + topic + " " + "emotion is: " + emotion);
        return String.format("inputText is: " + inputText + " " + "topic is: " + topic + " " + "emotion is: " + emotion);
//        mv.addObject("msg", "inputText is: " + inputText + " " + "category is: " + category + " " + "emotion is: " + emotion);
//        mv.setViewName("freemarker");
//        return mv;

//        return new Tweet(counter.incrementAndGet(), String.format(template, inputText, category, emotion));
    }

}
