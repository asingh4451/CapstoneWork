import 'package:flutter/material.dart';
import 'package:sampleproject1/pages/firstpage.dart';
import 'package:sampleproject1/pages/student.dart';
import 'package:sampleproject1/pages/admin.dart';
import 'package:sampleproject1/pages/teacher.dart';
void main() {
  runApp(MaterialApp(
    initialRoute:'/',
      routes: {
        '/':(context)=>firstPage(),
        '/student':(context)=>Student(),
        '/admin':(context)=>Admin(),
        '/teacher':(context)=>Teacher(),
      },
  ));
}