---
layout:     post
title:      "C++ 类的访问控制和继承 及三种继承方式的理解"
subtitle:   "chaitong"
date:       2020-04-09
author:     "CT"
header-img: "img/blog-bg.jpg"
tags:
    - C++
---

# 1. 类中的访问说明符
```cpp
class Parent{
public:
	...
private:
	...
protected:
	...
};
```
访问说明符| public |private |protected
--|--|--|--
类外用户  | √ |  × | ×
类内成员  | √ | √ | √
派生类成员 |√  |  × | √ 
友元 | √  | √  | √
# 2. 派生类的继承方式说明符

```cpp
class Child : public Parent {};
class Child : protected Parent {};
class Child : private Parent {};
```

### public继承方式：
- 基类中的**public**成员在派生类中仍为**public**；
- 基类中的**protected**成员在派生类中仍为**protected**；
- 基类中的**private**成员在派生类中**被继承下来**，但是**不可访问**；
### protected继承方式：
- 基类中的**public**成员在派生类中变为**protected**属性；
- 基类中的**protected**成员在派生类中变为**protected**属性；
- 基类中的**private**成员在派生类中**被继承下来**，但是仍**不可访问**；
### private继承方式：
- 基类中的**public**成员在派生类中变为**private**属性；
- 基类中的**protected**成员在派生类中变为**private**属性；
- 基类中的**private**成员在派生类中**被继承下来**，但是仍**不可访问**；
### 可以看出，三种继承方式不会影响派生类成员对基类成员的访问权限，无论哪种继承方式，派生类中仍然只能访问基类中的public和protected成员，不能访问private成员；
### 继承方式影响的是 类实例对象对类成员的访问权限：
- 例如，如果派生类的继承方式是private，则基类对象可以访问的public属性的类成员，派生类对象便不能访问了，因为派生类中该对象变成了private属性；
## 总结：继承方式决定了基类成员在派生类中的可见性，但不影响派生类对基类成员的访问权限