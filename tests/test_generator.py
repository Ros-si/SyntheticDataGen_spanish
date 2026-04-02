from src.errorGenerator import ErrorGenerator
from src import constants as c
import pytest
import pandas as pd
import spacy
import unittest
from unittest.mock import MagicMock, patch
from spacy.tokens import Doc

class TestErrorGenerator(unittest.TestCase):
    
    def setUp(self):
        self.nlp = spacy.load("es_core_news_lg")
        #self.nlp.remove_pipe("lemmatizer")
        #self.nlp.add_pipe("lemmatizer", config={"mode": "lookup"}).initialize()
       
        # Textos para probar los metodos generadores de errores

        self.texts_genre=[
        'estaba jugando en el jardín mientras el gato dormia.',
        'La niña encontró su mochila en la mesa.',
        'Una señora pidió ayuda porque ella no encontraba su boleto.',
        'intenta buscar consuelo en la comida'
        ]
        self.texts_numPlur=['El gato jugaba con los niños', 'está pintada de un color blanco','Me gusta tomar un café caliente','El sofá nuevo es muy cómodo','El jabalí corría rápidamente','la interfaz quedó muy buena','El mánager me pidió los resultados']
        self.texts_numSing=['Los gatos jugaban con el raton', 'Las casas del vecindario están de blanco','Me gusta tomar dos tazas de café caliente por la mañana','Las sillas son muy cómodas','Los caballos corrían rápidamente']
        self.texts_art=['El gato negro duerme en la silla.','Conoce el Estado de México.','El libro tiene historias muy interesantes','el cielo es azul.','la milla extra']
        self.texts_verbForm=['La niña dibujó un paisaje hermoso','Nosotros viajaremos en bus a la ciudad mañana.','Ellos comen en silencio.','caminas muy despacio.','estudiamos en la biblioteca todos los días.']
        self.texts_title=['La lujosa casa.','el niño.','mi hermana estudia en Harvard','los Emiratos Árabes']
        self.texts_accent=['La decisión final','Este examen será muy importante','Vió la novela que tanto esperaba']
        self.texts_punctuation=['No, me gusta la lluvia','¿Vas a la fiesta?', 'Pedro, llegas tarde.' ]
        self.texts_wo =[ 'comer manzanas todos los días es saludable.', # VERB NOUN
         'los felinos cazan eficientemente.', # VERB ADV
         'Zautla es uno de los 217 municipios que integran el estado mexicano de Puebla', #NOUN ADJ
         'Morton construyó una fuerte relación académica por correspondencia con Si', #ADJ NOUN
         'La tarea fue terminada por los estudiantes', # aux verb "Ser" + VERB
         'La comida está fría', # aux verb "Estar" + ADJ
         'Ellos están estudiando para los exámenes', # aux verb "Estar" + VERB
         'El resultado es positivo', # aux verb "Ser" + ADJ
         'Meses después de que Fagalde la propusiera, el 19 de febrero' # none
         ]
        self.texts_smistake=['El gato come pescado fresco.','ella visita a su abuela', 'estas helado',"salió por el garaje","la gigantesca montaña"]
        
        self.texts_nPlur_verbF=['No fumes mucho tabaco', 'Nuestra lámpara brilla mas en enero.']
        self.texts_vF_gen_wo=['intentaba buscar consuelo en la comida', 'la mañana del 2 de abril, jugaba feliz']

    def set_datafr(self, texts):
        datafr= pd.DataFrame(texts, columns=['sentence'])
        datafr['corrupted']=""
        datafr['tokens']= [list() for _ in range(len(datafr))]
        datafr['error_tags'] = [list() for _ in range(len(datafr))]
        datafr['error_type']= [list() for _ in range(len(datafr))]
        datafr['span'] = [list()for _ in range(len(datafr))]
        datafr['annotation'] = [list()for _ in range(len(datafr))]

        datafr['corrupted_tagged'] = ""
        datafr['aux_corrupted_tagged']=[list()for _ in range(len(datafr))]
        datafr['spaces']=[list()for _ in range(len(datafr))]
        return datafr
    
   
    def test_apply_error_genre(self):
        data_gen = self.set_datafr(self.texts_genre)
        errorGenerator = ErrorGenerator(data_gen, self.nlp)
        errorGenerator.apply_errors_ggenre(data_gen['sentence'],0)
        self.assertEqual(
            data_gen['corrupted'][0],
            'estaba jugando en la jardín mientras la gato dormia.'
        )
        self.assertEqual(
            data_gen['tokens'][0],
            ['estaba', 'jugando', 'en', 'la', 'jardín', 'mientras', 'la', 'gato', 'dormia', '.']
        )
        self.assertEqual(data_gen['error_tags'][0],[c.ErrorTag.NO_ERROR.id_num,
                                                    c.ErrorTag.NO_ERROR.id_num,
                                                    c.ErrorTag.NO_ERROR.id_num,
                                                    c.ErrorTag.GENDER.id_num,     # la
                                                    c.ErrorTag.NO_ERROR.id_num,
                                                    c.ErrorTag.NO_ERROR.id_num,
                                                    c.ErrorTag.GENDER.id_num,     # la
                                                    c.ErrorTag.NO_ERROR.id_num,
                                                    c.ErrorTag.NO_ERROR.id_num,
                                                    c.ErrorTag.NO_ERROR.id_num
                                                ])
        self.assertEqual(
            data_gen['corrupted'][1],'La niña encontró su mochila en el mesa.')
        self.assertEqual(data_gen['corrupted'][2], 'Una señora pidió ayuda porque él no encontraba su boleto.')
        self.assertEqual(data_gen['corrupted'][3], 'intenta buscar consuelo en el comida')


    def test_apply_error_gnumPlur(self):
        data = self.set_datafr(self.texts_numPlur)
        errorGenerator = ErrorGenerator(data, self.nlp, 3)
        errorGenerator.apply_errors_gnumPlur(data['sentence'],0)
        self.assertEqual(data['corrupted'][0],'El gatos jugaba con los niños')
        self.assertEqual(data['tokens'][0],['El', 'gatos', 'jugaba', 'con', 'los', 'niños'])
        self.assertEqual(data['error_tags'][0],[c.ErrorTag.NO_ERROR.id_num, 
                                                c.ErrorTag.NUM_PLUR.id_num, 
                                                c.ErrorTag.NO_ERROR.id_num, 
                                                c.ErrorTag.NO_ERROR.id_num, 
                                                c.ErrorTag.NO_ERROR.id_num, 
                                                c.ErrorTag.NO_ERROR.id_num])
        self.assertEqual(data['corrupted'][1],'está pintadas de un colores blancos')
        self.assertEqual(data['corrupted'][2],'Me gusta tomar un cafés calientes')
        self.assertEqual(data['corrupted'][3],'El sofás nuevos es muy cómodos')
        self.assertEqual(data['corrupted'][4],'El jabalís corría rápidamente')
        self.assertEqual(data['corrupted'][5],'la interfaces quedó muy buenas')
        self.assertEqual(data['corrupted'][6],'El mánager me pidió los resultados')
    

    def test_apply_error_gnumSing(self):
        data = self.set_datafr(self.texts_numSing)
        errorGenerator = ErrorGenerator(data, self.nlp)
        errorGenerator.apply_errors_gnumSing(data['sentence'],0)
        self.assertEqual(data['corrupted'][0],'Los gato jugaban con el raton')
        self.assertEqual(data['tokens'][0],['Los', 'gato', 'jugaban', 'con', 'el', 'raton'])
        self.assertEqual(data['error_tags'][0],[c.ErrorTag.NO_ERROR.id_num, c.ErrorTag.NUM_SING.id_num, c.ErrorTag.NO_ERROR.id_num, c.ErrorTag.NO_ERROR.id_num, c.ErrorTag.NO_ERROR.id_num, c.ErrorTag.NO_ERROR.id_num])
        self.assertEqual(data['corrupted'][1],'Las casa del vecindario están de blanco')
        self.assertEqual(data['corrupted'][2],'Me gusta tomar dos taza de café caliente por la mañana')
        self.assertEqual(data['corrupted'][3],'Las silla son muy cómodo')
        self.assertEqual(data['corrupted'][4],'Los caballo corrían rápidamente')
    
    def test_apply_error_guart(self):
        data = self.set_datafr(self.texts_art)
        errorGen = ErrorGenerator(data, self.nlp)
        errorGen.apply_errors_guart(data['sentence'],0)
        self.assertEqual(data['corrupted'][0],'gato negro duerme en silla.')
        self.assertEqual(data['tokens'][0],[' ','gato', 'negro', 'duerme','en',' ','silla','.'])
        self.assertEqual(data['error_tags'][0],[c.ErrorTag.ART_MISSING.id_num,
                                                c.ErrorTag.NO_ERROR.id_num,
                                                c.ErrorTag.NO_ERROR.id_num,
                                                c.ErrorTag.NO_ERROR.id_num,
                                                c.ErrorTag.NO_ERROR.id_num,
                                                c.ErrorTag.ART_MISSING.id_num,
                                                c.ErrorTag.NO_ERROR.id_num,
                                                c.ErrorTag.NO_ERROR.id_num,])
        self.assertIn(data['corrupted'][1],'Conoce Estado de México.')
        self.assertEqual(data['corrupted'][2],'libro tiene historias muy interesantes')
        self.assertEqual(data['corrupted'][3],'cielo es azul.')
        self.assertEqual(data['corrupted'][4],'milla extra')
    

    def test_apply_error_g_verbForm(self):
        data = self.set_datafr(self.texts_verbForm)
        errorGenerator = ErrorGenerator(data, self.nlp)
        errorGenerator.apply_errors_gverbForm(data['sentence'],0)
        self.assertEqual(data['corrupted'][0],'La niña dibujar un paisaje hermoso')
        self.assertEqual(data['error_tags'][0],[c.ErrorTag.NO_ERROR.id_num,
                                                c.ErrorTag.NO_ERROR.id_num,
                                                c.ErrorTag.VERB_FORM.id_num, 
                                                c.ErrorTag.NO_ERROR.id_num,
                                                c.ErrorTag.NO_ERROR.id_num,
                                                c.ErrorTag.NO_ERROR.id_num])
        self.assertEqual(data['tokens'][0],['La', 'niña', 'dibujar', 'un', 'paisaje' ,'hermoso'])
        self.assertEqual(data['corrupted'][1],'Nosotros viajar en bus a la ciudad mañana.')
        self.assertEqual(data['corrupted'][2],'Ellos comer en silencio.')
        self.assertEqual(data['corrupted'][3],'caminar muy despacio.')
        self.assertEqual(data['corrupted'][4],'estudiar en la biblioteca todos los días.')
      

    def test_apply_error_title(self):
        data = self.set_datafr(self.texts_title)
        errorGenerator = ErrorGenerator(data, self.nlp)
        errorGenerator.apply_errors_stitle(data['sentence'],0)        
        self.assertEqual(data['corrupted'][0],'La Lujosa Casa.')
        self.assertEqual(data['tokens'][0],['La', 'Lujosa','Casa','.'])
        self.assertEqual(data['error_tags'][0],[c.ErrorTag.NO_ERROR.id_num, 
                                                c.ErrorTag.TITLE.id_num, 
                                                c.ErrorTag.TITLE.id_num, 
                                                c.ErrorTag.NO_ERROR.id_num])
        self.assertEqual(data['corrupted'][1],'el Niño.')
        self.assertEqual(data['corrupted'][2],'mi Hermana Estudia en Harvard')
        self.assertEqual(data['corrupted'][3],'Los Emiratos Árabes')

   
    def test_apply_error_accent(self):
        data = self.set_datafr(self.texts_accent)
        errorGenerator = ErrorGenerator(data, self.nlp)
        errorGenerator.apply_errors_saccent(data['sentence'],0)
        self.assertEqual(data['corrupted'][0],'La decision final')
        self.assertEqual(data['tokens'][0],['La', 'decision', 'final'])
        self.assertEqual(data['error_tags'][0],[c.ErrorTag.NO_ERROR.id_num, 
                                                c.ErrorTag.ACCENT.id_num, 
                                                c.ErrorTag.NO_ERROR.id_num])
        self.assertEqual(data['corrupted'][1],'Este examen sera muy importante')
        self.assertEqual(data['corrupted'][2],'Vio la novela que tanto esperaba')
    
    def test_apply_error_ggword_order(self):        
        data_wo = self.set_datafr(self.texts_wo)
        errorGenerator = ErrorGenerator(data_wo, self.nlp)
        errorGenerator.apply_errors_ggword_order(data_wo['sentence'],0)
        self.assertEqual(data_wo['corrupted'][0],'manzanas comer todos los días saludable es.')
        self.assertEqual(len(data_wo['corrupted'][0]), len(data_wo['sentence'][0]))
        self.assertEqual(len(data_wo['tokens'][0]), len(data_wo['error_tags'][0]))
        self.assertEqual(data_wo['error_tags'][0],[c.ErrorTag.WORD_ORDER.id_num, 
                                                   c.ErrorTag.WORD_ORDER.id_num, 
                                                   c.ErrorTag.NO_ERROR.id_num, 
                                                   c.ErrorTag.NO_ERROR.id_num, 
                                                   c.ErrorTag.NO_ERROR.id_num ,
                                                   c.ErrorTag.WORD_ORDER.id_num, 
                                                   c.ErrorTag.WORD_ORDER.id_num, 
                                                   c.ErrorTag.NO_ERROR.id_num])
        self.assertEqual(data_wo['corrupted'][1],'los felinos eficientemente cazan.')
        self.assertEqual(data_wo['corrupted'][2],'Zautla es uno de los 217 municipios que integran el mexicano estado de Puebla')
        self.assertEqual(data_wo['corrupted'][3],'Morton construyó una relación fuerte académica por correspondencia con Si')
        self.assertEqual(len(data_wo['corrupted'][3]), len(data_wo['sentence'][3]))
        self.assertEqual(len(data_wo['tokens'][3]), len(data_wo['error_tags'][3]))
        self.assertEqual(data_wo['corrupted'][4],'La tarea terminada fue por los estudiantes')
        self.assertEqual(data_wo['corrupted'][5],'La comida fría está')
        self.assertEqual(data_wo['corrupted'][6],'Ellos estudiando están para los exámenes')
        self.assertEqual(data_wo['corrupted'][7],'El resultado positivo es')
        self.assertEqual(data_wo['corrupted'][8],'Meses después de que Fagalde la propusiera, el 19 de febrero')

    def test_apply_error_punctuation(self):
        data = self.set_datafr(self.texts_punctuation)
        errorGenerator = ErrorGenerator(data, self.nlp)
        errorGenerator.apply_errors_punctuation(data['sentence'],0)
        self.assertEqual(data['corrupted'][0],'No me gusta la lluvia')
        self.assertEqual(data['tokens'][0],['No',' ', 'me', 'gusta', 'la', 'lluvia'])
        self.assertEqual(data['error_tags'][0],[c.ErrorTag.NO_ERROR.id_num, 
                                                c.ErrorTag.PUNCT_MISSING.id_num, 
                                                c.ErrorTag.NO_ERROR.id_num, 
                                                c.ErrorTag.NO_ERROR.id_num, 
                                                c.ErrorTag.NO_ERROR.id_num,
                                                c.ErrorTag.NO_ERROR.id_num])
        self.assertIn(data['corrupted'][1],['Vas a la fiesta?','¿Vas a la fiesta','Vas a la fiesta'])
        self.assertIn(data['corrupted'][2],['Pedro llegas tarde.','Pedro, llegas tarde','Pedro llegas tarde'])    


    def test_apply_errors_vform_gen_wo(self):       
        data = self.set_datafr(self.texts_vF_gen_wo)
        errorGenerator = ErrorGenerator(data, self.nlp)
        errorGenerator.apply_errors_ggenre(data['sentence'],0)
        errorGenerator.apply_errors_gverbForm(data['corrupted'],0)        
        errorGenerator.apply_errors_ggword_order(data['corrupted'],0)
        self.assertEqual(data['corrupted'][0],'intentar consuelo buscar en el comida')
        self.assertEqual(data['corrupted'][1],'el mañana del 2 de abril, feliz jugar')
        self.assertEqual(len(data['tokens'][1]), len(data['error_tags'][1]))
        self.assertEqual(data['error_tags'][1],[c.ErrorTag.GENDER.id_num, 
                                                c.ErrorTag.NO_ERROR.id_num, 
                                                c.ErrorTag.NO_ERROR.id_num, 
                                                c.ErrorTag.NO_ERROR.id_num,
                                                c.ErrorTag.NO_ERROR.id_num, 
                                                c.ErrorTag.NO_ERROR.id_num, 
                                                c.ErrorTag.NO_ERROR.id_num, 
                                                c.ErrorTag.WORD_ORDER.id_num, 
                                                c.ErrorTag.WORD_ORDER.id_num])
        self.assertEqual(data['tokens'][1],['el', 'mañana', 'del', '2', 'de', 'abril',',', 'feliz', 'jugar'])
        self.assertEqual(data['corrupted_tagged'][1],'<G-gen el> mañana del 2 de abril, <G-wo feliz <G-verbForm jugar>>')

    #EG-MIX-03y 04
    def test_apply_errors_nplur_verbF(self):       
        data = self.set_datafr(self.texts_nPlur_verbF)
        errorGenerator = ErrorGenerator(data, self.nlp)
        errorGenerator.apply_errors_gnumPlur(data['sentence'],0)
        errorGenerator.apply_errors_gverbForm(data['corrupted'],0)
        self.assertEqual(data['corrupted'][0],'No fumar mucho tabacos')
        self.assertEqual(data['error_tags'][0], [c.ErrorTag.NO_ERROR.id_num, 
                                                 c.ErrorTag.VERB_FORM.id_num, 
                                                 c.ErrorTag.NO_ERROR.id_num, 
                                                 c.ErrorTag.NUM_PLUR.id_num])
        self.assertEqual(len(data['tokens'][0]), len(data['error_tags'][0]))
        self.assertEqual(data['corrupted_tagged'][0],'No <G-verbForm fumar> mucho <G-nPlur tabacos>')
        self.assertEqual(data['corrupted'][1],'Nuestra lámparas brillar mas en enero.')
        self.assertEqual(data['error_tags'][1], [c.ErrorTag.NO_ERROR.id_num, 
                                                 c.ErrorTag.NUM_PLUR.id_num, 
                                                 c.ErrorTag.VERB_FORM.id_num, 
                                                 c.ErrorTag.NO_ERROR.id_num, 
                                                 c.ErrorTag.NO_ERROR.id_num, 
                                                 c.ErrorTag.NO_ERROR.id_num, 
                                                 c.ErrorTag.NO_ERROR.id_num])


    def test_apply_spelling_mistake(self): 
        data= self.set_datafr(self.texts_smistake)
        errorGenerator = ErrorGenerator(data, self.nlp)
        errorGenerator.apply_errors_smistake(data['sentence'],0)
        self.assertEqual(data['corrupted'][0], 'El gato cone pezcado frezco.')
        self.assertEqual(data['error_tags'][0],[c.ErrorTag.NO_ERROR.id_num,
                                                 c.ErrorTag.NO_ERROR.id_num,
                                                 c.ErrorTag.SPELLING.id_num,
                                                 c.ErrorTag.SPELLING.id_num,
                                                 c.ErrorTag.SPELLING.id_num,
                                                 c.ErrorTag.NO_ERROR.id_num])
        self.assertIn(data['corrupted'][1], ['ella bizita a su avuela','ella vizita a su avuela','ella bisita a su avuela'])
        self.assertEqual(data['corrupted'][2], 'estas elado')
        self.assertEqual(data['corrupted'][3],'zalió por el garage')
        self.assertIn(data['corrupted'][4], ['la gigantezca nontaña','la gigamtesca nontaña','la jigantesca momtaña','la gigamtesca momtaña','la gigantesca nontaña','la jigantesca nontaña','la gigamtesca momtaña', 'la gigantezca momtaña'])
  
if __name__ == '__main__':
    unittest.main()
