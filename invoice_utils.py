
from pypdf import PdfReader
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAI

gpt4_api_version = "2023-07-01-preview"
gpt4_azure_api_key = "-----------------------------------------"
gpt4_azure_endpoint = "https://azure-openai-----------------azure.com/"
gpt4_deploy_name= "gpt-4"

llm_gpt4 = AzureChatOpenAI(temperature=0.0,
                                    model_name="gpt-4",
                                    openai_api_version=gpt4_api_version,
                                    azure_deployment=gpt4_deploy_name,
                                    openai_api_key=gpt4_azure_api_key,
                                    azure_endpoint=gpt4_azure_endpoint
                                )
# import neccessary packages from korr
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number

# iterate over files in
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list):
    """This function is used to extract the invoice data from the given PDF files. 
    It uses the LangChain agent to extract the data from the given PDF files."""
    df = pd.DataFrame({'Invoice no.': pd.Series(dtype='str'),
                   'Description': pd.Series(dtype='str'),
                   'Quantity': pd.Series(dtype='str'),
                   'Date': pd.Series(dtype='str'),
	                'Unit price': pd.Series(dtype='str'),
                   'Amount': pd.Series(dtype='int'),
                   'Total': pd.Series(dtype='str'),
                   'Email': pd.Series(dtype='str'),
	                'Phone number': pd.Series(dtype='str'),
                   'Address': pd.Series(dtype='str')
                    })

    for filename in user_pdf_list:

        # Extract PDF Data
        texts = ""
        print("Processing -", filename)
        pdf_reader = PdfReader(filename)
        for page in pdf_reader.pages:
            texts += page.extract_text()

        template = """Extract all the following values : invoice no., Description, Quantity, date, 
            Unit price , Amount, Total, email, phone number and address from the following Invoice content: 
            {texts}
            The fields and values in the above content may be jumbled up as they are extracted from a PDF. Please use your judgement to align
            the fields and values correctly based on the fields asked for in the question abiove.
            Expected output format: 
            {{'Invoice no.': xxxxxxxx','Description': 'xxxxxx', 'Quantity': 'x', 'Date': 'dd/mm/yyyy',
            'Unit price': xxx.xx','Amount': 'xxx.xx,'Total': xxx,xx,'Email': 'xxx@xxx.xxx','Phone number': 'xxxxxxxxxx','Address': 'xxxxxxxxx'}}
            Remove any dollar symbols or currency symbols from the extracted values.
            """
        prompt = PromptTemplate.from_template(template)


        chain = LLMChain(llm=llm_gpt4, prompt=prompt)

        data_dict = chain.run(texts)

        print("Dict:...", data_dict)
        new_row_df = pd.DataFrame([eval(data_dict)], columns=df.columns)
        df = pd.concat([df, new_row_df], ignore_index=True)  

        print("********************DONE***************")

    print(df) 
    return df

pdf= ["invoice_3452334.pdf", "invoice_3452334.pdf"]
create_docs(pdf) 