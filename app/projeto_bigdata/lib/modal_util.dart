import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

abstract class ModalUtil {

  static void exibirModalSucesso(BuildContext contexto, String mensagem) {
    var conteudo = Text(mensagem);
    var titulo = _titulo("Sucesso!", Icons.check);
    _exibirModal(contexto: contexto, titulo: titulo, conteudo: conteudo);
  }

  static void exibirModalAtencao(BuildContext contexto, String mensagem) {
    var conteudo = Text(mensagem);
    var titulo = _titulo("Atenção!", Icons.warning);
    _exibirModal(contexto: contexto, titulo: titulo, conteudo: conteudo);
  }

  static void exibirModalCarregando(BuildContext contexto) {
    var conteudo = Container(height: 100, width: 100, child: widgetEspera());
    var titulo = _titulo("Carregando...", Icons.sync);
    _exibirModal(contexto: contexto, titulo: titulo, conteudo: conteudo, permitirFechar: false);  
  }

  static void _exibirModal({BuildContext contexto, Widget titulo, Widget conteudo, bool permitirFechar = true}) {
    showGeneralDialog(
      barrierDismissible: permitirFechar,
      barrierColor: Colors.black.withOpacity(0.5),
      transitionBuilder: (context, a1, a2, widget) {
        return Transform.scale(
          scale: a1.value,
          child: Opacity(
            opacity: a1.value,
            child: WillPopScope(
              child: AlertDialog(
                title: titulo,
                content: conteudo,
              ),
              onWillPop: () => Future.value(permitirFechar),
            ),
          ),
        );
      },
      transitionDuration: Duration(milliseconds: 200),
      barrierLabel: '',
      context: contexto,
      pageBuilder: (context, animation1, animation2) {}
    );
    SystemChrome.setEnabledSystemUIOverlays([]);
  }

  static _titulo(String texto, IconData icone) {
    return Row(children: <Widget>[
      Icon(icone), 
      Padding(padding: EdgeInsets.only(left: 4)),
      Text(texto)
    ]);
  }

  static Widget widgetEspera() {
    return Center(
      heightFactor: 20,
      child: CircularProgressIndicator(
        backgroundColor: Colors.blueAccent,
        valueColor: AlwaysStoppedAnimation(Colors.white),
      ),
    );
  }
}
