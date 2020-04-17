import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:projeto_bigdata/modal_util.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(title: 'Reconhecimento de Desenhos'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key key, this.title}) : super(key: key);

  final String title;

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  Future<File> _tirarFoto() async {
    return await ImagePicker.pickImage(
      source: ImageSource.camera,
      maxHeight: 240.0,
      maxWidth: 240.0,
    );
  }

  void _enviarFoto() async {
    try {
      ModalUtil.exibirModalCarregando(context);

      var foto = await _tirarFoto();
      var request = http.MultipartRequest(
          'POST', Uri.parse('http://53d32f4c.ngrok.io/which_draw'));
      request.files.add(http.MultipartFile(
          'file', foto.readAsBytes().asStream(), foto.lengthSync(),
          filename: 'foto.png'));

      var response = await request.send();
      var responseBody = await response.stream.bytesToString();

      Navigator.of(context).pop();
      ModalUtil.exibirModalSucesso(context, responseBody);
    } on Exception catch (_) {
      Navigator.of(context).pop();
      ModalUtil.exibirModalAtencao(context, 'Erro inesperado!');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _enviarFoto,
        tooltip: 'Foto',
        child: Icon(Icons.camera_alt),
      ),
    );
  }
}
