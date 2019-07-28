### Mindmeld Conversion Tool

__Introduction__  
This tool is designed to help make the migration from other conversational AI platforms to MindMeld as seamless as possible. Mindmeld has many advantages that make it an ideal solution for building conversational assistants, and with this tool you can easily transfer your existing projects into MindMeld to expand their capabilities. Currently, we offer support for projects built in Rasa and Dialogflow.

__How does this tool work?__  
This tool will generate the basic framework of a MindMeld project, and will use the entities and intents of the existing project to populate the entities folders and create training files respectively. Dialogue state handlers will be created for each intent to handle any necessary logic behind responses, and a default configurations file will also be created, which may be changed as needed. If you are unfamiliar with the structure of MindMeld projects, code examples can be found in our MindMeld Blueprints [here](https://github.com/CiscoDevNet/mindmeld-blueprints/tree/develop/blueprints), and overviews can be found [here](https://www.mindmeld.com/docs/blueprints/overview.html).

__Usage__  
*Rasa Users:*  

*Dialogflow Users:*  
Users must first export their project out of Dialogflow. From the main console, click the __Settings__ icon next to your project name in the top left corner. From there select __Export and Import__, then __Export as Zip__. 
