package com.ca.group1.plp.pojo;

public class Tweet {

    private final long id;
    private final String content;

    public Tweet(long id, String content) {
        this.id = id;
        this.content = content;
    }

    public long getId() {
        return id;
    }

    public String getContent() {
        return content;
    }
}