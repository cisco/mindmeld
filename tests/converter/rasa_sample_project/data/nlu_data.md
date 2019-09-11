<!--- Make sure to update this training data file with more training examples from https://forum.rasa.com/t/rasa-starter-pack/704 --> 

## intent:goodbye <!--- The label of the intent --> 
- Bye 			<!--- Training examples for intent 'bye'--> 
- Goodbye
- See you later
- Bye bot
- Goodbye friend
- bye
- bye for now
- catch you later
- gotta go
- See you
- goodnight
- have a nice day
- i'm off
- see you later alligator
- we'll speak soon

## intent:greet
- Hi
- Hey
- Hi bot
- Hey bot
- Hello
- Good morning
- hi again
- hi folks
- hi Mister
- hi pal!
- hi there
- greetings
- hello everybody
- hello is anybody there
- hello robot

## intent:thanks
- Thanks
- Thank you
- Thank you so much
- Thanks bot
- Thanks for that
- cheers
- cheers bro
- ok thanks!
- perfect thank you
- thanks a bunch for everything
- thanks for the help
- thanks a lot
- amazing, thanks
- cool, thanks
- cool thank you

## intent:affirm
- yes
- yes sure
- absolutely
- for sure
- yes yes yes
- definitely

## intent:lastname
- My name is [Justin](name) [Andrews](lastname)  <!--- Square brackets contain the value of entity while the text in parentheses is a a label of the entity --> 
- I am [David](name) [Williams](lastname)
- I'm [Lacy](name) [Smith](lastname)
- People call me [Greg](name) [Harrding](lastname)
- It's [Brian](name) [Alley](lastname)
- Usually people call me [Amy](name) [Ardoin](lastname)
- My name is [Jacob](name) [Adams](lastname)
- You can call me [Mike](name) [Ellington](lastname)
- Please call me [Mary](name) [Ann](lastname)
- Name name is [Robert](name) [Goodman](lastname)
- I am [Richy](name) [Johnson](lastname)
- I'm [Tracy](name) [Jones](lastname)
- Call me [Sally](name) [Miller](lastname)
- I am [Philipp](name) [Wilson](lastname)
- I am [Charles](name) [Bennet](lastname)
- I am [Charles](name) [Brown](lastname)
- I am [Robby](name) [Clark](lastname)
- Call me [Susy](name) [Foster](lastname)

## intent:name
- My name is [Juste](name)  <!--- Square brackets contain the value of entity while the text in parentheses is a a label of the entity --> 
- I am [Josh](name)
- I'm [Lucy](name)
- People call me [Greg](name)
- It's [David](name)
- Usually people call me [Amy](name)
- My name is [John](name)
- You can call me [Sam](name)
- Please call me [Linda](name)
- Name name is [Tom](name)
- I am [Richard](name)
- I'm [Tracy](name)
- Call me [Sally](name)
- I am [Philipp](name)
- I am [Charlie](name)
- I am [Charlie](name)
- I am [Ben](name)
- Call me [Susan](name)
- [Lucy](name)
- [Peter](name)
- [Mark](name)
- [Joseph](name)
- [Tan](name)
- [Pete](name)
- [Elon](name)
- [Penny](name)
- [Barret](name)
- name is [Andrew](name)
- I [Lora](name)
- [Stan](name) is my name
- [Susan](name) is the name
- [Ross](name) is my first name
- [Bing](name) is my last name
- Few call me as [Angelina](name)
- Some call me [Julia](name)
- Everyone calls me [Laura](name)
- I am [Ganesh](name)
- My name is [Mike](name)
- just call me [Monika](name)
- Few call [Dan](name)
- You can always call me [Suraj](name)
- Some will call me [Andrew](name)
- My name is [Ajay](name)
- I call [Ding](name)
- I'm [Partia](name)
- Please call me [Leo](name)
- name is [Pari](name)
- name [Sanjay](name)

## intent:joke
- Can you tell me a joke?
- I would like to hear a joke
- Tell me a joke
- A joke please
- Tell me a joke please
- I would like to hear a joke
- I would loke to hear a joke, please
- Can you tell jokes?
- Please tell me a joke
- I need to hear a joke

## intent:deny
- No
- Not right now
- No thanks
- No thank you
- Don't do that
- Absolutely not
- No way
- definitely not
- Not in million years
