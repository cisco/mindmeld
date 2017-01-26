Workbench Application Directory Structure
=========================================

Describe the directory structure and organization for a workbench application repo::


  root/
      [vendor/]
      [scripts/]
      [ui/]
      [tests/]
      my_app.py
      domains/
          intent1/
          intent2/
      entities/
          global/
          entity1/
          entity2/
              entity_map.json
              entity_data.txt
              synonyms.txt
      qa/
          ranking.txt

Introduce the concept of a reference application, or 'blueprint', which is a pre-built set of code samples and data sets for the most typical application use cases. Show examples of the blueprint available for our 4 example applications: mobile ordering assistant, movie assistant, food delivery assistant, music discovery assistant.