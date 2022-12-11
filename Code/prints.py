'''
prints.py

Contains helper functions for printing things
'''

def small_banner(string, before_space = False, after_space = False):
    '''
    `small_banner`

    Prints something like:
    ############
    # <string> #
    ############

    and adds before/after spacing depending on inputs (False by default)
    '''
    if before_space:
        print()

    new_string = "# " + string + " #"
    length = len(new_string)
    print("#"*length)
    print(new_string)
    print("#"*length)

    if after_space:
        print()
